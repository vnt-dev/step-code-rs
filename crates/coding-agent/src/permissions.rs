//! Approval policy and shell command analysis.
//!
//! Ported functionally from `@step-harness/coding-agent`
//! `src/step/permissions.ts`, `src/step/command-policy.ts`, and
//! `src/step/shell-analysis.ts`.
//!
//! The TypeScript source drives an interactive confirmation dialog; this port
//! returns a [`Decision`] and lets the embedder render the prompt.

use std::collections::HashSet;

use std::sync::LazyLock;
use step_harness_providers::types::Json;

use crate::settings::PermissionPresetSetting;

/// Tools that never mutate the filesystem or execute code.
const READ_ONLY_TOOLS: &[&str] = &[
    "list_directory",
    "find_files",
    "search_files",
    "search_web",
    "read_file",
    "find_tools",
    "ls",
    "find",
    "grep",
    "read",
];

/// Tools that write files or execute commands.
const WRITE_OR_EXECUTE_TOOLS: &[&str] = &[
    "write_file",
    "edit_file",
    "run_command",
    "write",
    "edit",
    "bash",
    "powershell",
    "user_bash",
];

/// Wrapper commands unwrapped before classification.
const COMMAND_WRAPPERS: &[&str] = &[
    "sudo", "env", "doas", "nice", "nohup", "setsid", "stdbuf", "time", "timeout", "xargs", "exec",
    "command", "builtin",
];

/// Dangerous command rules, checked in order (first match wins).
struct DangerousRule {
    id: &'static str,
    matches: fn(&[String]) -> bool,
}

/// `rm -rf` (and short-flag variants).
fn is_recursive_force_remove(words: &[String]) -> bool {
    if words.first().map(|word| basename(word)) != Some("rm") {
        return false;
    }
    let mut recursive = false;
    let mut force = false;
    for word in words.iter().skip(1) {
        if word == "--" {
            break;
        }
        if word.starts_with("--") {
            if word == "--recursive" {
                recursive = true;
            }
            if word == "--force" {
                force = true;
            }
            continue;
        }
        if word.starts_with('-') && word.len() > 1 {
            for flag in word[1..].chars() {
                if flag == 'r' || flag == 'R' {
                    recursive = true;
                }
                if flag == 'f' {
                    force = true;
                }
            }
        }
    }
    recursive && force
}

/// System lifecycle commands (`reboot`, `systemctl reboot`, ...).
fn is_system_lifecycle(words: &[String]) -> bool {
    let Some(name) = words.first() else {
        return false;
    };
    let name = basename(name);
    if matches!(name, "reboot" | "shutdown") {
        return true;
    }
    if matches!(name, "init" | "loginctl" | "systemctl" | "telinit") {
        return words
            .iter()
            .skip(1)
            .any(|arg| matches!(arg.as_str(), "reboot" | "shutdown"));
    }
    false
}

/// `mkfs.*`.
fn is_format_filesystem(words: &[String]) -> bool {
    words
        .first()
        .map(|word| {
            let name = basename(word);
            name == "mkfs" || name.starts_with("mkfs.")
        })
        .unwrap_or(false)
}

/// `dd ... if=...`.
fn is_copy_device(words: &[String]) -> bool {
    words
        .first()
        .map(|word| basename(word) == "dd")
        .unwrap_or(false)
        && words.iter().any(|word| word.starts_with("if="))
}

/// `:> /dev/...` style truncation.
fn is_truncate_device(words: &[String]) -> bool {
    let truncates = words
        .first()
        .map(|word| word == ":" || word.starts_with(":>"))
        .unwrap_or(false);
    truncates && words.iter().any(|word| word.starts_with("/dev/"))
}

/// Destructive git operations.
fn is_destructive_git(words: &[String]) -> bool {
    if words.first().map(|word| basename(word)) != Some("git") {
        return false;
    }
    let args: Vec<&str> = words.iter().skip(1).map(String::as_str).collect();
    if args.len() >= 2 && args[0] == "reset" && args[1] == "--hard" {
        return true;
    }
    if args.len() >= 2 && args[0] == "clean" && args[1].starts_with('-') && args[1].contains('f') {
        return true;
    }
    if args.len() >= 2 && args[0] == "push" {
        return args[1..]
            .iter()
            .any(|arg| *arg == "--force" || *arg == "-f" || arg.starts_with("--force-with-lease"));
    }
    false
}

/// Destructive SQL.
fn is_destructive_sql(words: &[String]) -> bool {
    static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(?:drop\s+database|truncate\s+table)\b").unwrap()
    });
    let joined = words.join(" ");
    PATTERN.is_match(&joined)
}

const DANGEROUS_RULES: &[DangerousRule] = &[
    DangerousRule {
        id: "recursive-force-remove",
        matches: is_recursive_force_remove,
    },
    DangerousRule {
        id: "system-lifecycle",
        matches: is_system_lifecycle,
    },
    DangerousRule {
        id: "format-filesystem",
        matches: is_format_filesystem,
    },
    DangerousRule {
        id: "copy-device",
        matches: is_copy_device,
    },
    DangerousRule {
        id: "truncate-device",
        matches: is_truncate_device,
    },
    DangerousRule {
        id: "destructive-git",
        matches: is_destructive_git,
    },
    DangerousRule {
        id: "destructive-sql",
        matches: is_destructive_sql,
    },
];

/// Shell dialect of a command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellDialect {
    Bash,
    PowerShell,
}

/// Result of analyzing a command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandAnalysis {
    /// No dangerous rule matched and every word is literal.
    Ordinary,
    /// A dangerous rule matched.
    Matched { rule_id: &'static str },
    /// The script could not be fully analyzed.
    Unresolved { reason: String },
}

/// Outcome of a permission decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Allow,
    /// Ask the user; `hazardous` marks dangerous-command confirmations.
    Confirm {
        reason: String,
        hazardous: bool,
    },
    /// Block the call; `terminate` stops the current tool batch.
    Deny {
        reason: String,
        terminate: bool,
    },
}

/// The effective approval mode for a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Safe tools run; writes and commands require approval.
    Ask,
    /// Ordinary tools run; dangerous commands still require approval.
    Bypass,
    /// Full access: tool calls run without confirmation.
    Autopilot,
}

impl From<PermissionPresetSetting> for ApprovalMode {
    fn from(preset: PermissionPresetSetting) -> Self {
        match preset {
            PermissionPresetSetting::Ask => ApprovalMode::Ask,
            PermissionPresetSetting::Bypass => ApprovalMode::Bypass,
            PermissionPresetSetting::Autopilot => ApprovalMode::Autopilot,
        }
    }
}

impl ApprovalMode {
    /// Whether the mode auto-approves ordinary tool calls.
    pub fn is_auto(&self) -> bool {
        matches!(self, ApprovalMode::Bypass | ApprovalMode::Autopilot)
    }

    /// The stable label this mode is persisted under.
    pub fn as_str(&self) -> &'static str {
        match self {
            ApprovalMode::Ask => "ask",
            ApprovalMode::Bypass => "bypass",
            ApprovalMode::Autopilot => "autopilot",
        }
    }

    /// A persisted label, or `None` when it names no known mode.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "ask" => Some(ApprovalMode::Ask),
            "bypass" => Some(ApprovalMode::Bypass),
            "autopilot" => Some(ApprovalMode::Autopilot),
            _ => None,
        }
    }
}

/// Per-tool overrides layered above the preset.
#[derive(Clone, Debug, Default)]
pub struct ToolOverrides {
    /// Tool names always denied.
    pub deny: HashSet<String>,
    /// Tool names always allowed without confirmation.
    pub allow: HashSet<String>,
    /// Tool names always confirmed.
    pub confirm: HashSet<String>,
}

/// The approval policy for one session.
#[derive(Clone, Debug)]
pub struct PermissionPolicy {
    mode: ApprovalMode,
    overrides: ToolOverrides,
    /// Fallback when no UI can answer a confirmation request.
    non_interactive_approval: bool,
}

impl PermissionPolicy {
    /// Build a policy from a preset and overrides.
    pub fn new(
        mode: ApprovalMode,
        overrides: ToolOverrides,
        non_interactive_approval: bool,
    ) -> Self {
        PermissionPolicy {
            mode,
            overrides,
            non_interactive_approval,
        }
    }

    /// The effective mode.
    pub fn mode(&self) -> ApprovalMode {
        self.mode
    }

    /// Replace the mode (the UI can cycle presets at runtime).
    pub fn set_mode(&mut self, mode: ApprovalMode) {
        self.mode = mode;
    }

    /// Decide a tool call.
    ///
    /// Explicit denials apply in every mode. Full access skips confirmation
    /// rules; other modes analyze commands before applying tool overrides.
    pub fn decide(&self, tool_name: &str, args: &Json) -> Decision {
        // 1. Explicit deny override (outranks dangerous-command rules).
        if self.overrides.deny.contains(tool_name) {
            return Decision::Deny {
                reason: format!("tool \"{tool_name}\" is denied by policy"),
                terminate: true,
            };
        }

        // Full access never asks for tool or command approval.
        if self.mode == ApprovalMode::Autopilot {
            return Decision::Allow;
        }

        // 2. Dangerous-command rules.
        if let Some(command) = extract_command(tool_name, args) {
            let dialect = if tool_name == "powershell" {
                ShellDialect::PowerShell
            } else {
                ShellDialect::Bash
            };
            match analyze_command(&command, dialect) {
                CommandAnalysis::Matched { rule_id } => {
                    let reason = format!("dangerous command ({rule_id}): {command}");
                    return Decision::Confirm {
                        reason,
                        hazardous: true,
                    };
                }
                CommandAnalysis::Unresolved { reason } => {
                    let reason = format!("{reason}: {command}");
                    return Decision::Confirm {
                        reason,
                        hazardous: false,
                    };
                }
                CommandAnalysis::Ordinary => {}
            }
        }

        // 3. Explicit allow/confirm overrides.
        if self.overrides.allow.contains(tool_name) {
            return Decision::Allow;
        }
        if self.overrides.confirm.contains(tool_name) {
            return Decision::Confirm {
                reason: format!("tool \"{tool_name}\" requires confirmation by policy"),
                hazardous: false,
            };
        }

        let mutating = is_mutating(tool_name);

        // 5. Auto modes allow ordinary calls.
        if self.mode.is_auto() {
            return Decision::Allow;
        }

        // 6. Read-only tools always run.
        if !mutating {
            return Decision::Allow;
        }

        // 7. Everything else asks.
        Decision::Confirm {
            reason: format!("approve {tool_name}?"),
            hazardous: false,
        }
    }

    /// Whether a confirmation can be auto-resolved without a UI.
    pub fn can_auto_resolve(&self, decision: &Decision) -> bool {
        match decision {
            Decision::Confirm { hazardous, .. } => self.non_interactive_approval && !hazardous,
            _ => false,
        }
    }

    /// Resolve a confirmation without a UI, when policy permits.
    pub fn resolve_non_interactive(&self, decision: &Decision) -> Decision {
        if self.can_auto_resolve(decision) {
            Decision::Allow
        } else {
            decision.clone()
        }
    }
}

/// Whether a tool can mutate state. Unknown tools count as mutating.
pub fn is_mutating(tool_name: &str) -> bool {
    WRITE_OR_EXECUTE_TOOLS.contains(&tool_name) || !READ_ONLY_TOOLS.contains(&tool_name)
}

/// Extract the command string from a tool's arguments.
pub fn extract_command(tool_name: &str, args: &Json) -> Option<String> {
    if !matches!(
        tool_name,
        "bash" | "powershell" | "run_command" | "user_bash"
    ) {
        return None;
    }
    for key in ["command", "cmd", "script"] {
        if let Some(value) = args.get(key).and_then(Json::as_str) {
            return Some(value.to_owned());
        }
    }
    None
}

/// Analyze a command against the dangerous rules and the literal-word check.
pub fn analyze_command(command: &str, dialect: ShellDialect) -> CommandAnalysis {
    if dialect != ShellDialect::Bash {
        return CommandAnalysis::Unresolved {
            reason: "unsupported-shell".to_owned(),
        };
    }
    let segments = split_shell_segments(command);
    for segment in &segments {
        match analyze_segment(segment, 0) {
            CommandAnalysis::Ordinary => {}
            other => return other,
        }
    }
    CommandAnalysis::Ordinary
}

/// Split a script on unquoted control operators (`;`, `&&`, `||`, `|`, newline).
fn split_shell_segments(script: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut chars = script.chars().peekable();
    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => {
                current.push(ch);
                escaped = true;
            }
            '\'' | '"' => {
                if quote == Some(ch) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(ch);
                }
                current.push(ch);
            }
            ';' | '\n' if quote.is_none() => {
                segments.push(std::mem::take(&mut current));
            }
            '&' | '|' if quote.is_none() => {
                // Consume a doubled operator as one separator.
                if chars.peek() == Some(&ch) {
                    chars.next();
                }
                segments.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        segments.push(current);
    }
    segments
        .into_iter()
        .map(|segment| segment.trim().to_owned())
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// Analyze one pipeline segment: tokenize, unwrap wrappers, apply rules.
fn analyze_segment(segment: &str, depth: usize) -> CommandAnalysis {
    if depth > 8 {
        return CommandAnalysis::Unresolved {
            reason: "analysis-limit".to_owned(),
        };
    }
    let words = tokenize(segment);
    if words.is_empty() {
        return CommandAnalysis::Ordinary;
    }

    // Dynamic words make the segment unanalyzable.
    if words.iter().any(|word| is_dynamic(word)) {
        return CommandAnalysis::Unresolved {
            reason: "dynamic-command".to_owned(),
        };
    }

    // Unwrap wrappers (sudo/env/timeout/...).
    let mut words = words;
    loop {
        let Some(name) = words.first().map(|word| basename(word)) else {
            return CommandAnalysis::Ordinary;
        };
        if !COMMAND_WRAPPERS.contains(&name) {
            break;
        }
        // Drop the wrapper name and any of its flags; keep the wrapped command.
        let mut rest: Vec<String> = words.into_iter().skip(1).collect();
        while rest
            .first()
            .map(|word| word.starts_with('-'))
            .unwrap_or(false)
        {
            rest.remove(0);
        }
        words = rest;
    }

    // `bash -c '<script>'` re-parses the inner script.
    let name = basename(words.first().map(String::as_str).unwrap_or_default());
    if matches!(name, "bash" | "sh" | "zsh")
        && let Some(index) = words.iter().position(|word| word == "-c")
        && let Some(script) = words.get(index + 1)
    {
        return analyze_command(script, ShellDialect::Bash);
    }

    // Leading variable assignments (`FOO=bar cmd`) are skipped.
    let words: Vec<String> = words
        .into_iter()
        .skip_while(|word| word.contains('=') && !word.starts_with('-'))
        .collect();
    if words.is_empty() {
        return CommandAnalysis::Ordinary;
    }

    for rule in DANGEROUS_RULES {
        if (rule.matches)(&words) {
            return CommandAnalysis::Matched { rule_id: rule.id };
        }
    }
    CommandAnalysis::Ordinary
}

/// Tokenize a segment into words, honoring quotes.
fn tokenize(segment: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut has_word = false;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in segment.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => {
                current.push(ch);
                escaped = true;
            }
            '\'' | '"' => {
                if quote == Some(ch) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(ch);
                } else {
                    current.push(ch);
                }
            }
            c if c.is_whitespace() && quote.is_none() => {
                if has_word {
                    words.push(std::mem::take(&mut current));
                    has_word = false;
                }
            }
            c => {
                current.push(c);
                has_word = true;
            }
        }
    }
    if has_word {
        words.push(current);
    }
    words
}

/// A word is dynamic when it contains expansions or globs.
fn is_dynamic(word: &str) -> bool {
    word.contains('$') || word.contains('`') || word.contains('*') || word.contains('?')
}

/// Basename of a command path (`/usr/bin/rm` → `rm`).
fn basename(word: &str) -> &str {
    word.rsplit(['/', '\\']).next().unwrap_or(word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy(mode: ApprovalMode) -> PermissionPolicy {
        PermissionPolicy::new(mode, ToolOverrides::default(), false)
    }

    #[test]
    fn approval_mode_labels_round_trip() {
        for mode in [
            ApprovalMode::Ask,
            ApprovalMode::Bypass,
            ApprovalMode::Autopilot,
        ] {
            assert_eq!(ApprovalMode::from_label(mode.as_str()), Some(mode));
        }
        assert_eq!(ApprovalMode::from_label("plan"), None);
        assert_eq!(ApprovalMode::from_label(""), None);
    }

    #[test]
    fn detects_dangerous_commands() {
        for command in [
            "rm -rf /",
            "rm -fr build",
            "rm --recursive --force ./out",
            "sudo reboot",
            "systemctl shutdown",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git push --force origin main",
            "echo 'drop database x'",
        ] {
            assert_ne!(
                analyze_command(command, ShellDialect::Bash),
                CommandAnalysis::Ordinary,
                "expected {command} to be flagged"
            );
        }
    }

    #[test]
    fn ordinary_commands_pass() {
        for command in [
            "ls -la",
            "cargo test",
            "git status",
            "git push origin main",
            "rm file.txt",
            "cat README.md",
            "echo hello && echo world",
            "echo 'rm -rf /'",
        ] {
            assert_eq!(
                analyze_command(command, ShellDialect::Bash),
                CommandAnalysis::Ordinary,
                "expected {command} to be ordinary"
            );
        }
    }

    #[test]
    fn quoted_dangerous_commands_do_not_execute() {
        // The segment `echo 'rm -rf /'` is a single echo invocation.
        let analysis = analyze_command("echo 'rm -rf /'", ShellDialect::Bash);
        assert_eq!(analysis, CommandAnalysis::Ordinary);
    }

    #[test]
    fn dynamic_words_are_unresolved() {
        let analysis = analyze_command("rm -rf $TARGET", ShellDialect::Bash);
        assert!(matches!(analysis, CommandAnalysis::Unresolved { .. }));
    }

    #[test]
    fn wrapped_commands_are_unwrapped() {
        let analysis = analyze_command("sudo rm -rf /var/log/app", ShellDialect::Bash);
        assert_eq!(
            analysis,
            CommandAnalysis::Matched {
                rule_id: "recursive-force-remove"
            }
        );
    }

    #[test]
    fn nested_bash_c_is_analyzed() {
        let analysis = analyze_command("bash -c 'rm -rf /'", ShellDialect::Bash);
        assert_eq!(
            analysis,
            CommandAnalysis::Matched {
                rule_id: "recursive-force-remove"
            }
        );
    }

    #[test]
    fn ask_mode_confirms_mutations_and_allows_reads() {
        let ask = policy(ApprovalMode::Ask);
        assert_eq!(ask.decide("read", &json!({"path": "a"})), Decision::Allow);
        assert!(matches!(
            ask.decide("write", &json!({"path": "a"})),
            Decision::Confirm { .. }
        ));
        assert!(matches!(
            ask.decide("bash", &json!({"command": "ls"})),
            Decision::Confirm { .. }
        ));
    }

    #[test]
    fn autonomous_mode_confirms_dangerous_commands() {
        let bypass = policy(ApprovalMode::Bypass);
        let decision = bypass.decide("bash", &json!({"command": "rm -rf /"}));
        assert!(matches!(
            decision,
            Decision::Confirm {
                hazardous: true,
                ..
            }
        ));
        // Non-hazardous commands run in auto modes.
        assert_eq!(
            bypass.decide("bash", &json!({"command": "ls"})),
            Decision::Allow
        );
    }

    #[test]
    fn full_access_allows_commands_without_approval() {
        let full_access = policy(ApprovalMode::Autopilot);
        for (tool, command) in [
            ("bash", "rm -rf /"),
            ("bash", "rm -rf $TARGET"),
            ("powershell", "Remove-Item -Recurse ./build"),
            ("run_command", "git reset --hard HEAD~1"),
            ("user_bash", "reboot"),
        ] {
            assert_eq!(
                full_access.decide(tool, &json!({"command": command})),
                Decision::Allow,
                "full access should allow {tool}: {command}"
            );
        }
        for tool in ["read", "write", "edit", "some_future_tool"] {
            assert_eq!(full_access.decide(tool, &json!({})), Decision::Allow);
        }
    }

    #[test]
    fn full_access_skips_confirm_overrides_but_keeps_explicit_denials() {
        let mut overrides = ToolOverrides::default();
        overrides.confirm.insert("bash".to_owned());
        overrides.confirm.insert("write".to_owned());
        overrides.deny.insert("blocked_tool".to_owned());
        let full_access = PermissionPolicy::new(ApprovalMode::Autopilot, overrides, false);
        assert_eq!(
            full_access.decide("bash", &json!({"command": "rm -rf $TARGET"})),
            Decision::Allow
        );
        assert_eq!(full_access.decide("write", &json!({})), Decision::Allow);
        assert!(matches!(
            full_access.decide("blocked_tool", &json!({})),
            Decision::Deny {
                terminate: true,
                ..
            }
        ));
    }

    #[test]
    fn overrides_outrank_the_mutating_check() {
        let mut overrides = ToolOverrides::default();
        overrides.allow.insert("write".to_owned());
        let ask = PermissionPolicy::new(ApprovalMode::Ask, overrides, false);
        assert_eq!(ask.decide("write", &json!({"path": "a"})), Decision::Allow);
    }

    #[test]
    fn deny_overrides_outrank_dangerous_rules() {
        let mut overrides = ToolOverrides::default();
        overrides.deny.insert("bash".to_owned());
        let ask = PermissionPolicy::new(ApprovalMode::Ask, overrides, false);
        assert!(matches!(
            ask.decide("bash", &json!({"command": "rm -rf /"})),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn non_interactive_approval_covers_non_hazardous_confirmations_only() {
        let mut overrides = ToolOverrides::default();
        overrides.confirm.insert("write".to_owned());
        let lenient = PermissionPolicy::new(ApprovalMode::Ask, overrides, true);
        let decision = lenient.decide("write", &json!({"path": "a"}));
        assert_eq!(lenient.resolve_non_interactive(&decision), Decision::Allow);
        let strict = policy(ApprovalMode::Ask);
        assert!(matches!(
            strict.resolve_non_interactive(&Decision::Confirm {
                reason: "x".to_owned(),
                hazardous: true
            }),
            Decision::Confirm { .. }
        ));
    }

    #[test]
    fn powershell_commands_are_unresolved() {
        let analysis = analyze_command("Remove-Item -Recurse", ShellDialect::PowerShell);
        assert!(matches!(analysis, CommandAnalysis::Unresolved { .. }));
    }

    #[test]
    fn unknown_tools_are_mutating() {
        assert!(is_mutating("some_future_tool"));
        assert!(!is_mutating("read"));
        assert!(is_mutating("bash"));
    }
}

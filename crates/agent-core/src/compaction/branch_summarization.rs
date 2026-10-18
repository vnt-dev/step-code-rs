//! Branch summarization: summarizing an abandoned branch before the
//! conversation navigates elsewhere in the session tree.
//!
//! Ported from `@step-harness/agent-core`
//! `src/harness/compaction/branch-summarization.ts`.
//!
//! Notable differences from the TypeScript source:
//! - `models.completeSimple` is replaced by [`SimpleCompleter`], carried on
//!   [`GenerateBranchSummaryOptions`] together with the [`Model`].
//! - The TypeScript `Result<T, BranchSummaryError>` monad is
//!   [`std::result::Result`]; session failures surface as [`SessionError`]
//!   instead of a thrown `SessionError`.
//! - Token counts are `u64`; the `reserveTokens` default of 16384 is exposed as
//!   [`DEFAULT_BRANCH_RESERVE_TOKENS`] and applied by
//!   [`GenerateBranchSummaryOptions::new`].
//! - Tool-result messages are skipped when collecting branch content, unlike
//!   the compaction pipeline which keeps them.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{Context, Model, SimpleStreamOptions, StopReason, Usage};
use step_harness_providers::utils::retry::{RetryCallbacks, RetryPolicy};
use step_harness_providers::utils::text::assistant_content_text;

use crate::compaction::SimpleCompleter;
use crate::compaction::core::{
    complete_simple_with_retries, estimate_tokens, summarization_user_message,
};
use crate::compaction::utils::{
    FileOperations, SUMMARIZATION_SYSTEM_PROMPT, compute_file_lists, create_file_ops,
    extract_file_ops_from_message, format_file_operations, serialize_conversation,
};
use crate::harness::messages::{
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
};
use crate::harness::types::{AbortSignal, BranchSummaryError, BranchSummaryErrorCode};
use crate::session::facade::Session;
use crate::session::types::{
    BranchBounds, BranchEntryQuery, Entry, SessionError, SessionErrorCode,
};
use crate::types::AgentMessage;

/// Default tokens reserved for prompt and model output when generating a
/// branch summary.
pub const DEFAULT_BRANCH_RESERVE_TOKENS: u64 = 16384;

/// Generated branch summary data ready to be persisted as a branch-summary entry.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchSummaryResult {
    /// Summary text, including the branch preamble and file-operation tags.
    pub summary: String,
    /// Usage from the summarization request.
    pub usage: Option<Usage>,
    /// Files read while exploring the summarized branch.
    pub read_files: Vec<String>,
    /// Files modified while exploring the summarized branch.
    pub modified_files: Vec<String>,
}

/// File-operation details stored on generated branch summary entries.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryDetails {
    /// Files read while exploring the summarized branch.
    pub read_files: Vec<String>,
    /// Files modified while exploring the summarized branch.
    pub modified_files: Vec<String>,
}

/// Prepared branch content for summarization.
#[derive(Clone, Debug)]
pub struct BranchPreparation {
    /// Messages selected for the branch summary.
    pub messages: Vec<AgentMessage>,
    /// File operations extracted from the branch.
    pub file_ops: FileOperations,
    /// Estimated token count for selected messages.
    pub total_tokens: u64,
}

/// Entries selected for branch summarization.
#[derive(Clone, Debug)]
pub struct CollectEntriesResult {
    /// Entries to summarize in chronological order.
    pub entries: Vec<Entry>,
    /// Deepest common ancestor between the previous leaf and target entry.
    pub common_ancestor_id: Option<String>,
}

/// Options for generating a branch summary.
pub struct GenerateBranchSummaryOptions<'a> {
    /// Provider collection the summarization request goes through; owns auth resolution.
    pub completer: &'a dyn SimpleCompleter,
    /// Model used for summarization.
    pub model: &'a Model,
    /// Abort signal for the summarization request.
    pub signal: Option<AbortSignal>,
    /// Optional instructions appended to or replacing the default prompt.
    pub custom_instructions: Option<&'a str>,
    /// Replace the default prompt with custom instructions instead of appending them.
    pub replace_instructions: bool,
    /// Tokens reserved for prompt and model output.
    ///
    /// Defaults to [`DEFAULT_BRANCH_RESERVE_TOKENS`]; use
    /// [`GenerateBranchSummaryOptions::new`] to get the documented defaults.
    pub reserve_tokens: u64,
    /// Optional retry policy for transient summarization errors.
    pub retry: Option<&'a RetryPolicy>,
    /// Optional callbacks for retry reporting.
    pub callbacks: Option<&'a (dyn RetryCallbacks + Send + Sync)>,
}

impl<'a> GenerateBranchSummaryOptions<'a> {
    /// Options with the documented defaults: no signal, no custom
    /// instructions, `reserve_tokens` of [`DEFAULT_BRANCH_RESERVE_TOKENS`], and
    /// no retry policy.
    pub fn new(completer: &'a dyn SimpleCompleter, model: &'a Model) -> Self {
        GenerateBranchSummaryOptions {
            completer,
            model,
            signal: None,
            custom_instructions: None,
            replace_instructions: false,
            reserve_tokens: DEFAULT_BRANCH_RESERVE_TOKENS,
            retry: None,
            callbacks: None,
        }
    }
}

/// Collect entries that should be summarized before navigating to a different session tree entry.
pub async fn collect_entries_for_branch_summary<M>(
    session: &Session<M>,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> Result<CollectEntriesResult, SessionError>
where
    M: Clone + Send + Sync + 'static,
{
    let Some(old_leaf_id) = old_leaf_id else {
        return Ok(CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        });
    };
    let old_path: BTreeSet<String> = session
        .find_entries_on_branch(&BranchEntryQuery {
            bounds: BranchBounds {
                start: Some(old_leaf_id.to_owned()),
                ..Default::default()
            },
            ..Default::default()
        })
        .await?
        .into_iter()
        .map(|entry| entry.id().to_owned())
        .collect();
    let target_path = session
        .find_entries_on_branch(&BranchEntryQuery {
            bounds: BranchBounds {
                start: Some(target_id.to_owned()),
                ..Default::default()
            },
            ..Default::default()
        })
        .await?;
    let mut common_ancestor_id: Option<String> = None;
    for entry in &target_path {
        if old_path.contains(entry.id()) {
            common_ancestor_id = Some(entry.id().to_owned());
            break;
        }
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut current: Option<String> = Some(old_leaf_id.to_owned());

    while let Some(id) = current {
        if common_ancestor_id.as_deref() == Some(id.as_str()) {
            break;
        }
        let Some(entry) = session.get_entry(&id).await else {
            return Err(SessionError::new(
                SessionErrorCode::InvalidEntry,
                format!("Entry {id} not found"),
            ));
        };
        current = entry.parent_id().map(str::to_owned);
        entries.push(entry);
    }
    entries.reverse();

    Ok(CollectEntriesResult {
        entries,
        common_ancestor_id,
    })
}

fn get_message_from_entry(entry: &Entry) -> Option<AgentMessage> {
    match entry {
        Entry::Message(message) => {
            // Tool results carry no standalone narrative for a branch summary.
            if matches!(message.message, AgentMessage::ToolResult(_)) {
                return None;
            }
            Some(message.message.clone())
        }
        Entry::BranchSummary(summary) => {
            Some(AgentMessage::BranchSummary(create_branch_summary_message(
                summary.summary.clone(),
                summary.from_id.clone(),
                summary.timestamp,
            )))
        }
        Entry::Compaction(compaction) => Some(AgentMessage::CompactionSummary(
            create_compaction_summary_message(
                compaction.summary.clone(),
                compaction.tokens_before,
                compaction.timestamp,
            ),
        )),
        _ => None,
    }
}

/// Prepare branch entries for summarization within an optional token budget.
///
/// A `token_budget` of `0` means "no budget": every entry is kept.
pub fn prepare_branch_entries(entries: &[Entry], token_budget: u64) -> BranchPreparation {
    let mut file_ops = create_file_ops();
    for entry in entries {
        if let Entry::BranchSummary(summary) = entry
            && let Some(details) = &summary.details
            && let Ok(details) = serde_json::from_value::<BranchSummaryDetails>(details.clone())
        {
            file_ops.read.extend(details.read_files);
            file_ops.edited.extend(details.modified_files);
        }
    }

    // Walking newest-first and reversing at the end keeps the chronological
    // order of the TypeScript `unshift` loop without its quadratic copies.
    let mut collected: Vec<AgentMessage> = Vec::new();
    let mut total_tokens: u64 = 0;
    for entry in entries.iter().rev() {
        let Some(message) = get_message_from_entry(entry) else {
            continue;
        };
        extract_file_ops_from_message(&message, &mut file_ops);

        let tokens = estimate_tokens(&message);
        if token_budget > 0 && total_tokens.saturating_add(tokens) > token_budget {
            if matches!(entry, Entry::Compaction(_) | Entry::BranchSummary(_))
                && u128::from(total_tokens) * 10 < u128::from(token_budget) * 9
            {
                collected.push(message);
                total_tokens = total_tokens.saturating_add(tokens);
            }
            break;
        }

        collected.push(message);
        total_tokens = total_tokens.saturating_add(tokens);
    }
    collected.reverse();

    BranchPreparation {
        messages: collected,
        file_ops,
        total_tokens,
    }
}

pub const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

pub const BRANCH_SUMMARY_PROMPT: &str = r#"Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."#;

/// Generate a summary for abandoned branch entries.
pub async fn generate_branch_summary(
    entries: &[Entry],
    options: &GenerateBranchSummaryOptions<'_>,
) -> Result<BranchSummaryResult, BranchSummaryError> {
    let GenerateBranchSummaryOptions {
        completer,
        model,
        signal,
        custom_instructions,
        replace_instructions,
        reserve_tokens,
        retry,
        callbacks,
    } = options;
    let context_window = if model.context_window > 0 {
        model.context_window
    } else {
        128_000
    };
    let token_budget = context_window.saturating_sub(*reserve_tokens);

    let BranchPreparation {
        messages,
        file_ops,
        total_tokens: _,
    } = prepare_branch_entries(entries, token_budget);

    if messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_owned(),
            usage: None,
            read_files: Vec::new(),
            modified_files: Vec::new(),
        });
    }
    let llm_messages = convert_to_llm(&messages);
    let conversation_text = serialize_conversation(&llm_messages, None);
    let instructions = match (custom_instructions, *replace_instructions) {
        (Some(instructions), true) => (*instructions).to_owned(),
        (Some(instructions), false) => {
            format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {instructions}")
        }
        (None, _) => BRANCH_SUMMARY_PROMPT.to_owned(),
    };
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");

    let context = Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![summarization_user_message(prompt_text)],
        tools: None,
    };
    // Branch summaries use a fixed output budget and no reasoning request,
    // matching the TypeScript `{ signal, maxTokens: 2048 }` options.
    let mut request_options = SimpleStreamOptions::default();
    request_options.base.max_tokens = Some(2048);
    request_options.base.base.signal = signal.clone();

    let response = complete_simple_with_retries(
        *completer,
        model,
        &context,
        &request_options,
        *retry,
        *callbacks,
    )
    .await;
    if response.stop_reason == StopReason::Aborted {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::Aborted,
            response
                .error_message
                .as_deref()
                .unwrap_or("Branch summary aborted"),
        ));
    }
    if response.stop_reason == StopReason::Error {
        return Err(BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!(
                "Branch summary failed: {}",
                response.error_message.as_deref().unwrap_or("Unknown error")
            ),
        ));
    }

    let mut summary = format!(
        "{BRANCH_SUMMARY_PREAMBLE}{}",
        assistant_content_text(&response.content, "\n")
    );
    let (read_files, modified_files) = compute_file_lists(&file_ops);
    summary.push_str(&format_file_operations(&read_files, &modified_files));
    // The preamble is a non-empty constant, so this fallback is unreachable in
    // practice; it preserves the TypeScript `summary || "No summary generated"`.
    let summary = if summary.is_empty() {
        "No summary generated".to_owned()
    } else {
        summary
    };

    Ok(BranchSummaryResult {
        summary,
        usage: Some(response.usage),
        read_files,
        modified_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::memory::InMemorySessionStorage;
    use crate::session::types::{ProvisionedEntry, SessionMetadata};
    use step_harness_providers::types::{
        Api, AssistantMessage, CacheRetention, ProviderId, TextContent, UserContentBlock,
    };

    fn session() -> Session<SessionMetadata> {
        let storage = std::sync::Arc::new(InMemorySessionStorage::new(SessionMetadata {
            id: "s1".to_owned(),
            created_at: 1,
            parent_session_id: None,
        }));
        Session::new(storage, None)
    }

    fn assistant_with_tool_call(path: &str) -> AgentMessage {
        AgentMessage::Assistant(Box::new(AssistantMessage {
            content: vec![step_harness_providers::types::AssistantContent::ToolCall(
                step_harness_providers::types::ToolCall {
                    id: "c1".to_owned(),
                    name: "read".to_owned(),
                    arguments: serde_json::json!({ "path": path })
                        .as_object()
                        .unwrap()
                        .clone(),
                    thought_signature: None,
                    namespace: None,
                },
            )],
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 1,
        }))
    }

    fn tool_result_entry(id: &str, seq: u64, parent_id: Option<String>) -> Entry {
        ProvisionedEntry::Message {
            id: id.to_owned(),
            message: AgentMessage::tool_result(step_harness_providers::types::ToolResultMessage {
                tool_call_id: "c1".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![UserContentBlock::Text(TextContent {
                    text: "body".to_owned(),
                    text_signature: None,
                })],
                details: None,
                usage: None,
                added_tool_names: None,
                is_error: false,
                timestamp: seq,
            }),
            terminate: false,
        }
        .into_entry(parent_id, seq, seq * 10)
    }

    fn message_entry(
        id: &str,
        seq: u64,
        parent_id: Option<String>,
        message: AgentMessage,
    ) -> Entry {
        ProvisionedEntry::Message {
            id: id.to_owned(),
            message,
            terminate: false,
        }
        .into_entry(parent_id, seq, seq * 10)
    }

    fn model() -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: Vec::new(),
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 0,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    /// Completer that is never expected to run in these tests.
    struct NoopCompleter;

    #[async_trait::async_trait]
    impl SimpleCompleter for NoopCompleter {
        async fn complete_simple(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &SimpleStreamOptions,
        ) -> AssistantMessage {
            empty_assistant_message()
        }
    }

    /// Completer that answers every request with a fixed text block.
    struct PrefilledCompleter {
        text: String,
    }

    #[async_trait::async_trait]
    impl SimpleCompleter for PrefilledCompleter {
        async fn complete_simple(
            &self,
            _model: &Model,
            _context: &Context,
            options: &SimpleStreamOptions,
        ) -> AssistantMessage {
            let mut message = empty_assistant_message();
            message.content = vec![step_harness_providers::types::AssistantContent::Text(
                TextContent {
                    text: self.text.clone(),
                    text_signature: None,
                },
            )];
            // Branch summaries use a fixed output budget and no reasoning request.
            assert_eq!(options.base.max_tokens, Some(2048));
            assert!(options.reasoning.is_none());
            // Summaries are isolated: no cache retention and a fresh session id.
            assert_eq!(options.base.cache_retention, Some(CacheRetention::None));
            assert!(options.base.session_id.is_some());
            message
        }
    }

    fn empty_assistant_message() -> AssistantMessage {
        AssistantMessage {
            content: Vec::new(),
            api: Api::AnthropicMessages,
            provider: ProviderId::new("test"),
            model: "m".to_owned(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Default::default(),
            stop_reason: StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn collect_entries_returns_empty_without_an_old_leaf() {
        let session = session();
        let result = collect_entries_for_branch_summary(&session, None, "whatever")
            .await
            .unwrap();
        assert!(result.entries.is_empty());
        assert_eq!(result.common_ancestor_id, None);
    }

    #[tokio::test]
    async fn collect_entries_walks_back_to_the_common_ancestor() {
        let session = session();
        let second = session
            .append_message(AgentMessage::user("two", 2))
            .await
            .unwrap();
        let third = session
            .append_message(AgentMessage::user("three", 3))
            .await
            .unwrap();
        // A divergent branch hanging off `second`.
        session.create_lane("alt", Some(&second)).await.unwrap();
        let divergent = session
            .view("alt")
            .append_message(AgentMessage::user("diverged", 4))
            .await
            .unwrap();

        let result = collect_entries_for_branch_summary(&session, Some(&divergent), &third)
            .await
            .unwrap();
        assert_eq!(result.common_ancestor_id.as_deref(), Some(second.as_str()));
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].id(), divergent);
    }

    #[tokio::test]
    async fn collect_entries_propagates_a_missing_target_error() {
        let session = session();
        let second = session
            .append_message(AgentMessage::user("two", 2))
            .await
            .unwrap();
        // Navigating to an unknown id fails in the branch walk.
        let result =
            collect_entries_for_branch_summary(&session, Some(&second), "missing-target").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, SessionErrorCode::NotFound);
    }

    #[test]
    fn prepare_branch_entries_skips_tool_results_and_keeps_order() {
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("first", 1)),
            tool_result_entry("t1", 2, Some("u1".to_owned())),
            message_entry(
                "u2",
                3,
                Some("t1".to_owned()),
                AgentMessage::user("second", 2),
            ),
        ];
        let preparation = prepare_branch_entries(&entries, 0);
        assert_eq!(preparation.messages.len(), 2);
        assert!(matches!(preparation.messages[0], AgentMessage::User(_)));
        assert!(matches!(preparation.messages[1], AgentMessage::User(_)));
        assert_eq!(preparation.messages[0].timestamp(), 1);
        assert_eq!(preparation.messages[1].timestamp(), 2);
        // "first" (2 tokens) + "second" (2 tokens).
        assert_eq!(preparation.total_tokens, 4);
    }

    #[test]
    fn prepare_branch_entries_respects_the_token_budget() {
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("a".repeat(400), 1)),
            message_entry(
                "u2",
                2,
                Some("u1".to_owned()),
                AgentMessage::user("b".repeat(400), 2),
            ),
            message_entry(
                "u3",
                3,
                Some("u2".to_owned()),
                AgentMessage::user("tail", 3),
            ),
        ];
        // A 101-token budget keeps the two newest messages (100 + 1 tokens).
        let preparation = prepare_branch_entries(&entries, 101);
        assert_eq!(preparation.messages.len(), 2);
        assert_eq!(preparation.total_tokens, 101);
        // No budget keeps everything.
        assert_eq!(prepare_branch_entries(&entries, 0).messages.len(), 3);
    }

    #[test]
    fn prepare_branch_entries_seeds_file_ops_from_branch_summary_details() {
        let branch_summary = ProvisionedEntry::BranchSummary {
            id: "b1".to_owned(),
            from_id: "f1".to_owned(),
            summary: "earlier".to_owned(),
            details: Some(serde_json::json!({
                "readFiles": ["read.rs"],
                "modifiedFiles": ["changed.rs"]
            })),
            usage: None,
        }
        .into_entry(None, 1, 10);
        let entries = vec![
            branch_summary,
            message_entry(
                "a1",
                2,
                Some("b1".to_owned()),
                assistant_with_tool_call("new.rs"),
            ),
        ];
        let preparation = prepare_branch_entries(&entries, 0);
        assert!(preparation.file_ops.read.contains("read.rs"));
        assert!(preparation.file_ops.read.contains("new.rs"));
        assert!(preparation.file_ops.edited.contains("changed.rs"));
        let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
        assert_eq!(read_files, vec!["new.rs".to_owned(), "read.rs".to_owned()]);
        assert_eq!(modified_files, vec!["changed.rs".to_owned()]);
    }

    #[test]
    fn branch_prompts_are_ported_verbatim() {
        assert_eq!(
            BRANCH_SUMMARY_PREAMBLE,
            "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n"
        );
        assert!(
            BRANCH_SUMMARY_PROMPT
                .starts_with("Create a structured summary of this conversation branch")
        );
        assert!(BRANCH_SUMMARY_PROMPT.contains("## Goal"));
        assert!(BRANCH_SUMMARY_PROMPT.contains("## Key Decisions"));
        assert!(BRANCH_SUMMARY_PROMPT.ends_with("Keep each section concise. Preserve exact file paths, function names, and error messages."));
    }

    #[test]
    fn default_options_carry_the_documented_reserve() {
        let completer = NoopCompleter;
        let model = model();
        let options = GenerateBranchSummaryOptions::new(&completer, &model);
        assert_eq!(options.reserve_tokens, DEFAULT_BRANCH_RESERVE_TOKENS);
        assert_eq!(options.reserve_tokens, 16384);
        assert!(!options.replace_instructions);
        assert!(options.custom_instructions.is_none());
        assert!(options.signal.is_none());
        assert!(options.retry.is_none());
    }

    #[tokio::test]
    async fn generate_branch_summary_reports_no_content_without_messages() {
        let completer = NoopCompleter;
        let model = model();
        let options = GenerateBranchSummaryOptions::new(&completer, &model);
        // Only tool-result entries: nothing to summarize.
        let entries = vec![tool_result_entry("t1", 1, None)];
        let result = generate_branch_summary(&entries, &options).await.unwrap();
        assert_eq!(result.summary, "No content to summarize");
        assert!(result.read_files.is_empty());
        assert!(result.modified_files.is_empty());
        assert!(result.usage.is_none());
    }

    #[tokio::test]
    async fn generate_branch_summary_wraps_the_model_reply_with_the_preamble() {
        let completer = PrefilledCompleter {
            text: "## Goal\nfinish it".to_owned(),
        };
        let model = model();
        let mut options = GenerateBranchSummaryOptions::new(&completer, &model);
        options.custom_instructions = Some("focus on tests");
        let entries = vec![
            message_entry("u1", 1, None, AgentMessage::user("do the thing", 1)),
            message_entry(
                "a1",
                2,
                Some("u1".to_owned()),
                assistant_with_tool_call("src.rs"),
            ),
        ];
        let result = generate_branch_summary(&entries, &options).await.unwrap();
        assert!(result.summary.starts_with(BRANCH_SUMMARY_PREAMBLE));
        assert!(result.summary.contains("## Goal\nfinish it"));
        assert!(
            result
                .summary
                .ends_with("<read-files>\nsrc.rs\n</read-files>")
        );
        assert_eq!(result.read_files, vec!["src.rs".to_owned()]);
        assert!(result.modified_files.is_empty());
    }
}

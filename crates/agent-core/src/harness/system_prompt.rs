//! System-prompt rendering for the model-visible skill list.
//!
//! Ported from `@step-harness/agent-core` `src\harness\system-prompt.ts`.

use crate::harness::types::Skill;

/// Render the `<available_skills>` block for a system prompt.
///
/// Skills with `disableModelInvocation` set are excluded. Returns an empty
/// string when no skills are visible.
pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible_skills: Vec<&Skill> = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect();
    if visible_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        "The following skills provide specialized instructions for specific tasks.".to_owned(),
        "Read the full skill file when the task matches its description.".to_owned(),
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_owned(),
        String::new(),
        "<available_skills>".to_owned(),
    ];

    for skill in visible_skills {
        lines.push("  <skill>".to_owned());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path)
        ));
        lines.push("  </skill>".to_owned());
    }

    lines.push("</available_skills>".to_owned());
    lines.join("\n")
}

/// Escape the five XML entities, in the TypeScript source's replacement order.
///
/// The order is load-bearing: `&` must be escaped first, otherwise the `&`
/// characters introduced by later replacements would be double-escaped.
fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &str, description: &str, file_path: &str) -> Skill {
        Skill {
            name: name.to_owned(),
            description: description.to_owned(),
            content: String::new(),
            file_path: file_path.to_owned(),
            disable_model_invocation: false,
        }
    }

    #[test]
    fn returns_empty_string_without_visible_skills() {
        assert_eq!(format_skills_for_system_prompt(&[]), "");
        let mut hidden = skill("pdf", "Reads PDFs", "/skills/pdf/SKILL.md");
        hidden.disable_model_invocation = true;
        assert_eq!(format_skills_for_system_prompt(&[hidden]), "");
    }

    #[test]
    fn renders_available_skills_block() {
        let skills = vec![skill("pdf", "Reads PDFs", "/skills/pdf/SKILL.md")];
        let rendered = format_skills_for_system_prompt(&skills);
        assert!(rendered.starts_with(
            "The following skills provide specialized instructions for specific tasks."
        ));
        assert!(rendered.contains("<available_skills>"));
        assert!(rendered.contains("    <name>pdf</name>"));
        assert!(rendered.contains("    <description>Reads PDFs</description>"));
        assert!(rendered.contains("    <location>/skills/pdf/SKILL.md</location>"));
        assert!(rendered.ends_with("</available_skills>"));
    }

    #[test]
    fn skips_model_invocation_disabled_skills() {
        let mut hidden = skill("secret", "Hidden", "/skills/secret/SKILL.md");
        hidden.disable_model_invocation = true;
        let visible = skill("pdf", "Reads PDFs", "/skills/pdf/SKILL.md");
        let rendered = format_skills_for_system_prompt(&[hidden, visible]);
        assert!(!rendered.contains("secret"));
        assert!(rendered.contains("pdf"));
    }

    #[test]
    fn escapes_xml_entities_in_source_order() {
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
        // `&` first: later escapes must not be double-escaped.
        assert_eq!(escape_xml("&<>\"'"), "&amp;&lt;&gt;&quot;&apos;");
    }

    #[test]
    fn escapes_skill_fields() {
        let skills = vec![skill(
            "a&b",
            "desc <with> \"quotes\" & 'apostrophes'",
            "/skills/a&b/SKILL.md",
        )];
        let rendered = format_skills_for_system_prompt(&skills);
        assert!(rendered.contains("<name>a&amp;b</name>"));
        assert!(rendered.contains("<description>desc &lt;with&gt; &quot;quotes&quot; &amp; &apos;apostrophes&apos;</description>"));
        assert!(rendered.contains("<location>/skills/a&amp;b/SKILL.md</location>"));
    }
}

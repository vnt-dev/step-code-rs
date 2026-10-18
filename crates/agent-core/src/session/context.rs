//! LLM context construction from session entries.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\context.ts`.

use std::sync::Arc;

use step_harness_providers::types::Usage;

use crate::harness::messages::{create_branch_summary_message, create_compaction_summary_message};
use crate::session::types::*;
use crate::types::AgentMessage;

/// The context derived from a branch of session entries.
#[derive(Clone, Debug, Default)]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    pub model: Option<SessionModel>,
    pub active_tool_names: Option<Vec<String>>,
}

/// The model a branch of entries was produced with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionModel {
    pub provider: String,
    pub model_id: String,
}

/// A transform applied to the entries that feed the context.
pub type ContextEntryTransform = Arc<dyn Fn(&[Entry]) -> Vec<Entry> + Send + Sync>;

/// A projector for custom entries into context messages.
pub type CustomEntryContextMessageProjector =
    Arc<dyn Fn(&CustomEntry, usize, &[Entry]) -> Option<Vec<AgentMessage>> + Send + Sync>;

/// Options for [`build_session_context`].
#[derive(Clone, Default)]
pub struct SessionContextBuildOptions {
    pub entry_transforms: Vec<ContextEntryTransform>,
    pub entry_projectors: std::collections::HashMap<String, CustomEntryContextMessageProjector>,
}

fn derive_session_context_state(
    path_entries: &[Entry],
) -> (String, Option<SessionModel>, Option<Vec<String>>) {
    let mut thinking_level = "off".to_owned();
    let mut model: Option<SessionModel> = None;
    let mut active_tool_names: Option<Vec<String>> = None;

    for entry in path_entries {
        match entry {
            Entry::ThinkingLevelChange(change) => thinking_level = change.thinking_level.clone(),
            Entry::ModelChange(change) => {
                model = Some(SessionModel {
                    provider: change.provider.clone(),
                    model_id: change.model_id.clone(),
                })
            }
            Entry::Message(message) => {
                if let AgentMessage::Assistant(assistant) = &message.message {
                    model = Some(SessionModel {
                        provider: assistant.provider.as_str().to_owned(),
                        model_id: assistant.model.clone(),
                    })
                }
            }
            Entry::ActiveToolsChange(change) => {
                active_tool_names = Some(change.active_tool_names.clone())
            }
            _ => {}
        }
    }
    (thinking_level, model, active_tool_names)
}

/// Keep only the last compaction entry and everything after it.
pub fn default_context_entry_transform(path_entries: &[Entry]) -> Vec<Entry> {
    let Some(compaction_index) = path_entries
        .iter()
        .rposition(|entry| matches!(entry, Entry::Compaction(_)))
    else {
        return path_entries.to_vec();
    };
    let mut entries = vec![path_entries[compaction_index].clone()];
    entries.extend_from_slice(&path_entries[compaction_index + 1..]);
    entries
}

/// Apply the default transform plus any custom transforms.
pub fn build_context_entries(
    path_entries: &[Entry],
    options: &SessionContextBuildOptions,
) -> Vec<Entry> {
    let mut entries = default_context_entry_transform(path_entries);
    for transform in &options.entry_transforms {
        entries = transform(&entries);
    }
    entries
}

/// Project one entry into context messages.
pub fn session_entry_to_context_messages(
    entry: &Entry,
    index: usize,
    entries: &[Entry],
    options: &SessionContextBuildOptions,
) -> Vec<AgentMessage> {
    match entry {
        Entry::Message(message) => {
            if matches!(&message.message, AgentMessage::Assistant(assistant) if assistant.stop_reason == step_harness_providers::types::StopReason::Deferred)
            {
                return Vec::new();
            }
            vec![message.message.clone()]
        }
        Entry::Compaction(compaction) => {
            let mut messages = vec![AgentMessage::CompactionSummary(
                create_compaction_summary_message(
                    compaction.summary.clone(),
                    compaction.tokens_before,
                    compaction.timestamp,
                ),
            )];
            messages.extend(compaction.retained_tail.iter().cloned());
            messages
        }
        Entry::BranchSummary(summary) => {
            if summary.summary.is_empty() {
                return Vec::new();
            }
            vec![AgentMessage::BranchSummary(create_branch_summary_message(
                summary.summary.clone(),
                summary.from_id.clone(),
                summary.timestamp,
            ))]
        }
        Entry::Custom(custom) => options
            .entry_projectors
            .get(&custom.custom_type)
            .and_then(|projector| projector(custom, index, entries))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Build the LLM context for a branch of entries.
pub fn build_session_context(
    path_entries: &[Entry],
    options: &SessionContextBuildOptions,
) -> SessionContext {
    let (thinking_level, model, active_tool_names) = derive_session_context_state(path_entries);
    let context_entries = build_context_entries(path_entries, options);
    let mut messages = Vec::new();
    for (index, entry) in context_entries.iter().enumerate() {
        messages.extend(session_entry_to_context_messages(
            entry,
            index,
            &context_entries,
            options,
        ));
    }
    SessionContext {
        messages,
        thinking_level,
        model,
        active_tool_names,
    }
}

/// Re-export used by compaction helpers.
pub type ContextUsage = Usage;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::types::ProvisionedEntry;
    use crate::types::AgentMessage;

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

    #[test]
    fn keeps_only_the_last_compaction_and_after() {
        let compaction = ProvisionedEntry::Compaction {
            id: "c1".to_owned(),
            summary: "old".to_owned(),
            retained_tail: vec![AgentMessage::user("kept", 5)],
            tokens_before: 100,
            details: None,
            usage: None,
        }
        .into_entry(None, 1, 10);
        let before = message_entry("e1", 2, None, AgentMessage::user("before", 1));
        let after = message_entry(
            "e2",
            3,
            Some("c1".to_owned()),
            AgentMessage::user("after", 2),
        );
        // Without a compaction entry everything is kept.
        assert_eq!(
            default_context_entry_transform(&[before.clone(), after.clone()]).len(),
            2
        );
        // Only the compaction entry and what follows it survive.
        let transformed = default_context_entry_transform(&[before, compaction, after]);
        assert_eq!(transformed.len(), 2);
        assert!(matches!(transformed[0], Entry::Compaction(_)));
    }

    #[test]
    fn projects_compaction_entries_into_summary_messages() {
        let compaction = ProvisionedEntry::Compaction {
            id: "c1".to_owned(),
            summary: "the summary".to_owned(),
            retained_tail: vec![AgentMessage::user("tail", 5)],
            tokens_before: 100,
            details: None,
            usage: None,
        }
        .into_entry(None, 1, 10);
        let context = build_session_context(&[compaction], &SessionContextBuildOptions::default());
        assert_eq!(context.messages.len(), 2);
        assert!(matches!(
            context.messages[0],
            AgentMessage::CompactionSummary(_)
        ));
        assert!(matches!(context.messages[1], AgentMessage::User(_)));
    }

    #[test]
    fn derives_state_from_entries() {
        let entries = vec![
            message_entry("m1", 1, None, AgentMessage::user("hi", 1)),
            ProvisionedEntry::ThinkingLevelChange {
                id: "t1".to_owned(),
                thinking_level: "high".to_owned(),
            }
            .into_entry(Some("m1".to_owned()), 2, 20),
            ProvisionedEntry::ModelChange {
                id: "mo1".to_owned(),
                provider: "p".to_owned(),
                model_id: "m".to_owned(),
            }
            .into_entry(Some("t1".to_owned()), 3, 30),
        ];
        let context = build_session_context(&entries, &SessionContextBuildOptions::default());
        assert_eq!(context.thinking_level, "high");
        assert_eq!(
            context.model,
            Some(SessionModel {
                provider: "p".to_owned(),
                model_id: "m".to_owned()
            })
        );
    }
}

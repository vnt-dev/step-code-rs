//! Harness custom message roles and their projection into LLM messages.
//!
//! Ported from `@step-harness/agent-core` `src/harness/messages.ts`.

use serde::{Deserialize, Serialize};
use step_harness_providers::types::{
    Message, TextContent, UserContent, UserContentBlock, UserMessage,
};

use crate::types::AgentMessage;

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

/// A recorded bash execution, projected as a user message for the LLM.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub command: String,
    pub output: String,
    /// Exit code; `None` when the command was killed by a signal.
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub exclude_from_context: bool,
}

/// Application-defined custom message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub custom_type: String,
    pub content: CustomMessageContent,
    pub display: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Content of a [`CustomMessage`]: plain text or mixed text/image blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomMessageContent {
    Text(String),
    Blocks(Vec<UserContentBlock>),
}

/// Summary of a branch the conversation came back from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    pub summary: String,
    pub from_id: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Summary injected in place of compacted history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub summary: String,
    /// Context tokens before compaction.
    pub tokens_before: u64,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Render a bash execution message as transcript text.
pub fn bash_execution_to_text(message: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", message.command);
    if message.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str("```\n");
        text.push_str(&message.output);
        text.push_str("\n```");
    }
    if message.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = message.exit_code
        && code != 0
    {
        text.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    if message.truncated
        && let Some(path) = &message.full_output_path
    {
        text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
    }
    text
}

/// Create a branch summary message. `timestamp` is Unix milliseconds.
pub fn create_branch_summary_message(
    summary: impl Into<String>,
    from_id: impl Into<String>,
    timestamp: u64,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        summary: summary.into(),
        from_id: from_id.into(),
        timestamp,
    }
}

/// Create a compaction summary message. `timestamp` is Unix milliseconds.
pub fn create_compaction_summary_message(
    summary: impl Into<String>,
    tokens_before: u64,
    timestamp: u64,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        summary: summary.into(),
        tokens_before,
        timestamp,
    }
}

/// Create a custom message. `timestamp` is Unix milliseconds.
pub fn create_custom_message(
    custom_type: impl Into<String>,
    content: impl Into<CustomMessageContent>,
    display: bool,
    details: Option<serde_json::Value>,
    timestamp: u64,
) -> CustomMessage {
    CustomMessage {
        custom_type: custom_type.into(),
        content: content.into(),
        display,
        details,
        timestamp,
    }
}

impl From<String> for CustomMessageContent {
    fn from(value: String) -> Self {
        CustomMessageContent::Text(value)
    }
}

impl From<&str> for CustomMessageContent {
    fn from(value: &str) -> Self {
        CustomMessageContent::Text(value.to_owned())
    }
}

impl From<Vec<UserContentBlock>> for CustomMessageContent {
    fn from(value: Vec<UserContentBlock>) -> Self {
        CustomMessageContent::Blocks(value)
    }
}

/// Convert agent messages to LLM-compatible messages.
///
/// `bashExecution` becomes a user text message (unless excluded), `custom`
/// becomes a user message, `branchSummary`/`compactionSummary` become user
/// messages with their prefix/suffix wrappers, and standard LLM messages pass
/// through unchanged. Custom roles without a projection are dropped.
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<Message> {
    let mut converted = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            AgentMessage::User(user) => converted.push(Message::User(user.clone())),
            AgentMessage::Assistant(assistant) => {
                converted.push(Message::Assistant(assistant.clone()))
            }
            AgentMessage::ToolResult(result) => converted.push(Message::ToolResult(result.clone())),
            AgentMessage::BashExecution(bash) => {
                if bash.exclude_from_context {
                    continue;
                }
                converted.push(Message::User(UserMessage {
                    content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                        text: bash_execution_to_text(bash),
                        text_signature: None,
                    })]),
                    timestamp: bash.timestamp,
                }));
            }
            AgentMessage::Custom(custom) => {
                let content = match &custom.content {
                    CustomMessageContent::Text(text) => {
                        UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                            text: text.clone(),
                            text_signature: None,
                        })])
                    }
                    CustomMessageContent::Blocks(blocks) => UserContent::Blocks(blocks.clone()),
                };
                converted.push(Message::User(UserMessage {
                    content,
                    timestamp: custom.timestamp,
                }));
            }
            AgentMessage::BranchSummary(summary) => {
                converted.push(Message::User(UserMessage {
                    content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                        text: format!(
                            "{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}",
                            summary.summary
                        ),
                        text_signature: None,
                    })]),
                    timestamp: summary.timestamp,
                }));
            }
            AgentMessage::CompactionSummary(summary) => {
                converted.push(Message::User(UserMessage {
                    content: UserContent::Blocks(vec![UserContentBlock::Text(TextContent {
                        text: format!(
                            "{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}",
                            summary.summary
                        ),
                        text_signature: None,
                    })]),
                    timestamp: summary.timestamp,
                }));
            }
        }
    }
    converted
}

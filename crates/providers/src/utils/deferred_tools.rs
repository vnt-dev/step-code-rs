//! Deferred tool loading: split current tools into a request-time prefix and
//! transcript-loaded definitions.

use std::collections::{HashMap, HashSet};

use crate::types::{Context, Tool};

/// Result of [`split_deferred_tools`]: tools sent with the request and tools
/// that the transcript loads later (keyed by normalized name).
pub struct SplitTools {
    pub immediate: Vec<Tool>,
    pub deferred: HashMap<String, Tool>,
}

/// Split current tools into prefix and transcript-loaded definitions.
///
/// A tool is deferred when a tool-result message announces it via
/// `added_tool_names` and no earlier assistant message already called it.
pub fn split_deferred_tools(
    context: &Context,
    enabled: bool,
    normalize_name: impl Fn(&str) -> String,
) -> SplitTools {
    let mut unique_tools: HashMap<String, Tool> = HashMap::new();
    if let Some(tools) = &context.tools {
        for tool in tools {
            unique_tools.insert(normalize_name(&tool.name), tool.clone());
        }
    }
    if !enabled {
        return SplitTools {
            immediate: unique_tools.into_values().collect(),
            deferred: HashMap::new(),
        };
    }

    let mut deferred_names: HashSet<String> = HashSet::new();
    let mut used_names: HashSet<String> = HashSet::new();
    for message in &context.messages {
        match message {
            crate::types::Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let Some(call) = block.tool_call() {
                        used_names.insert(normalize_name(&call.name));
                    }
                }
            }
            crate::types::Message::ToolResult(result) => {
                for name in result.added_tool_names.iter().flatten() {
                    let normalized = normalize_name(name);
                    if !used_names.contains(&normalized) {
                        deferred_names.insert(normalized);
                    }
                }
            }
            crate::types::Message::User(_) => {}
        }
    }

    let mut immediate = Vec::new();
    let mut deferred = HashMap::new();
    for (name, tool) in unique_tools {
        if deferred_names.contains(&name) {
            deferred.insert(name, tool);
        } else {
            immediate.push(tool);
        }
    }
    SplitTools {
        immediate,
        deferred,
    }
}

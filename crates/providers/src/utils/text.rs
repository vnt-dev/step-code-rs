//! Text helpers over message content blocks.

use crate::types::{AssistantContent, UserContent, UserContentBlock};

/// Extract and join text from message content.
pub fn content_text(content: &UserContent, separator: &str) -> String {
    match content {
        UserContent::Text(text) => text.clone(),
        UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(UserContentBlock::text)
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join(separator),
    }
}

/// Extract and join text from assistant content blocks.
pub fn assistant_content_text(content: &[AssistantContent], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| block.text().map(|text| text.text.as_str()))
        .collect::<Vec<_>>()
        .join(separator)
}

//! GitHub Copilot dynamic request headers.

use crate::types::{Message, UserContent};

/// Copilot expects `X-Initiator` to indicate whether the request is
/// user-initiated or agent-initiated (follow-up after assistant/tool messages).
pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(message) if !matches!(message, Message::User(_)) => "agent",
        _ => "user",
    }
}

/// Copilot requires `Copilot-Vision-Request` when sending images.
pub fn has_copilot_vision_input(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::User(user) => user
            .content
            .blocks()
            .is_some_and(|blocks| blocks.iter().any(|block| block.image().is_some())),
        Message::ToolResult(result) => result.content.iter().any(|block| block.image().is_some()),
        _ => false,
    })
}

/// Build Copilot's per-request dynamic headers.
pub fn build_copilot_dynamic_headers(
    messages: &[Message],
    has_images: bool,
) -> Vec<(String, String)> {
    let mut headers = vec![
        (
            "X-Initiator".to_owned(),
            infer_copilot_initiator(messages).to_owned(),
        ),
        ("Openai-Intent".to_owned(), "conversation-edits".to_owned()),
    ];
    if has_images {
        headers.push(("Copilot-Vision-Request".to_owned(), "true".to_owned()));
    }
    headers
}

#[allow(unused)]
fn unused_guard(content: &UserContent) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{UserContentBlock, UserMessage};

    #[test]
    fn initiator_depends_on_last_message_role() {
        let user_only = vec![Message::User(UserMessage {
            content: UserContent::Text("hi".to_owned()),
            timestamp: 1,
        })];
        assert_eq!(infer_copilot_initiator(&user_only), "user");
    }

    #[test]
    fn vision_detected_from_image_blocks() {
        let messages = vec![Message::User(UserMessage {
            content: UserContent::Blocks(vec![UserContentBlock::Image(
                crate::types::ImageContent {
                    data: "aaaa".to_owned(),
                    mime_type: "image/png".to_owned(),
                },
            )]),
            timestamp: 1,
        })];
        assert!(has_copilot_vision_input(&messages));
        let headers = build_copilot_dynamic_headers(&messages, true);
        assert!(headers.contains(&("Copilot-Vision-Request".to_owned(), "true".to_owned())));
        assert!(headers.contains(&("X-Initiator".to_owned(), "user".to_owned())));
    }
}

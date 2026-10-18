//! Model-generated names for the first prompt of an unnamed session.

use std::{
    sync::{Arc, Weak, atomic::Ordering},
    time::Duration,
};

use step_harness_agent_core::{compaction::SimpleCompleter, session::Entry, types::AgentMessage};
use step_harness_providers::types::{
    Context, Message, Model, SimpleStreamOptions, StopReason, StreamOptions, UserContent,
    UserContentBlock, UserMessage,
};
use tokio_util::sync::CancellationToken;

use super::{CatalogCompleter, SessionShared, events::SessionEvent};

const NAME_TIMEOUT: Duration = Duration::from_secs(30);
const NAME_TOKEN_BUDGET: u64 = 1024;
const NAME_ATTEMPTS: u32 = 2;
const MAX_NAME_CHARS: usize = 50;
const MAX_PROMPT_CHARS: usize = 3000;
pub(super) const MANUAL_NAME_ENTRY: &str = "session_manual_name";

pub(super) struct NameRequest {
    revision: u64,
    message: UserMessage,
}

pub(super) async fn prepare_name_request(
    shared: &SessionShared,
    message: &AgentMessage,
) -> Option<NameRequest> {
    if shared.name_generation_active.load(Ordering::SeqCst) {
        return None;
    }
    let revision = shared.name_revision.lock().await;
    if *revision != 0
        || shared
            .session
            .get_name()
            .await
            .is_some_and(|name| !name.trim().is_empty())
    {
        return None;
    }
    // A later user message can retry a failed title, but always summarizes the
    // original first message, including after reopening an unnamed session.
    let entries = shared
        .session
        .find_entries_on_branch(&super::chronological_branch_query())
        .await
        .ok()?;
    if entries.iter().any(
        |entry| matches!(entry, Entry::Custom(entry) if entry.custom_type == MANUAL_NAME_ENTRY),
    ) {
        return None;
    }
    let mut message = entries
        .iter()
        .find_map(|entry| match entry {
            Entry::Message(entry) => entry.message.as_user().cloned(),
            _ => None,
        })
        .or_else(|| message.as_user().cloned())?;
    match &mut message.content {
        UserContent::Text(text) => *text = text.chars().take(MAX_PROMPT_CHARS).collect(),
        UserContent::Blocks(blocks) => {
            for block in blocks {
                if let UserContentBlock::Text(text) = block {
                    text.text = text.text.chars().take(MAX_PROMPT_CHARS).collect();
                }
            }
        }
    }
    Some(NameRequest {
        revision: *revision,
        message,
    })
}

pub(super) fn generate_name(shared: &Arc<SessionShared>, model: Model, request: NameRequest) {
    if shared.name_generation_active.swap(true, Ordering::SeqCst) {
        return;
    }
    let completer: Arc<dyn SimpleCompleter> =
        shared.services.name_completer.clone().unwrap_or_else(|| {
            Arc::new(CatalogCompleter {
                models: shared.models_handle.clone(),
                observer: shared.services.request_observer.clone(),
                request_context: step_harness_providers::request_metrics::RequestContext { session_id: shared.metadata.id.clone(), execution_id: None, purpose: "naming".into() },
            })
        });
    let weak = Arc::downgrade(shared);
    tokio::spawn(async move {
        let _generation_guard = GenerationGuard(weak.clone());
        let token = CancellationToken::new();
        let _cancel_on_drop = token.clone().drop_guard();
        let mut options = SimpleStreamOptions {
            base: StreamOptions {
                max_tokens: Some(NAME_TOKEN_BUDGET),
                ..Default::default()
            },
            ..Default::default()
        };
        options.base.base.signal = Some(token);
        let context = Context {
            system_prompt: Some("You are a title generator. You output ONLY a thread title. Nothing else.\n\n<task>\nGenerate a brief title that would help the user find this conversation later.\n\nFollow all rules in <rules>\nUse the <examples> so you know what a good title looks like.\nYour output must be:\n- A single line\n- ≤50 characters\n- No explanations\n</task>\n\n<rules>\n- you MUST use the same language as the user message you are summarizing\n- Title must be grammatically correct and read naturally - no word salad\n- Never include tool names in the title (e.g. \"read tool\", \"bash tool\", \"edit tool\")\n- Focus on the main topic or question the user needs to retrieve\n- Vary your phrasing - avoid repetitive patterns like always starting with \"Analyzing\"\n- When a file is mentioned, focus on WHAT the user wants to do WITH the file, not just that they shared it\n- Keep exact: technical terms, numbers, filenames, HTTP codes\n- Remove: the, this, my, a, an\n- Never assume tech stack\n- Never use tools\n- NEVER respond to questions, just generate a title for the conversation\n- The title should NEVER include \"summarizing\" or \"generating\" when generating a title\n- DO NOT SAY YOU CANNOT GENERATE A TITLE OR COMPLAIN ABOUT THE INPUT\n- Always output something meaningful, even if the input is minimal.\n- If the user message is short or conversational (e.g. \"hello\", \"lol\", \"what's up\", \"hey\"):\n  → create a title that reflects the user's tone or intent (such as Greeting, Quick check-in, Light chat, Intro message, etc.)\n</rules>\n\n<examples>\n\"debug 500 errors in production\" → Debugging production 500 errors\n\"refactor user service\" → Refactoring user service\n\"why is app.js failing\" → app.js failure investigation\n\"implement rate limiting\" → Rate limiting implementation\n\"how do I connect postgres to my API\" → Postgres API connection\n\"best practices for React hooks\" → React hooks best practices\n\"@src/auth.ts can you add refresh token support\" → Auth refresh token support\n\"@utils/parser.ts this is broken\" → Parser bug fix\n\"look at @config.json\" → Config review\n\"@App.tsx add dark mode toggle\" → Dark mode toggle in App\n</examples>\n".to_owned()),
            messages: vec![Message::User(request.message)],
            tools: None,
        };
        let mut generated_name = None;
        for attempt in 0..NAME_ATTEMPTS {
            options.base.max_tokens = Some(NAME_TOKEN_BUDGET * u64::from(attempt + 1));
            match tokio::time::timeout(
                NAME_TIMEOUT,
                completer.complete_simple(&model, &context, &options),
            )
            .await
            {
                Ok(response) if response.stop_reason == StopReason::Stop => {
                    let text = response
                        .content
                        .iter()
                        .filter_map(|block| block.text())
                        .map(|text| text.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    if let Some(name) = normalize_name(&text) {
                        generated_name = Some(name);
                        break;
                    }
                    log::warn!(target: "session_name", "model={} attempt={} returned empty title", model.id, attempt + 1);
                }
                Ok(response) => {
                    log::warn!(target: "session_name", "model={} attempt={} stop_reason={:?}", model.id, attempt + 1, response.stop_reason)
                }
                Err(_) => {
                    log::warn!(target: "session_name", "model={} attempt={} timed out", model.id, attempt + 1)
                }
            }
            if let Some(shared) = weak.upgrade() {
                if *shared.name_revision.lock().await != request.revision {
                    return;
                }
            } else {
                return;
            }
        }
        let Some(name) = generated_name else {
            return;
        };
        let Some(shared) = weak.upgrade() else {
            return;
        };
        let revision = shared.name_revision.lock().await;
        if *revision != request.revision
            || shared
                .session
                .get_name()
                .await
                .is_some_and(|name| !name.trim().is_empty())
        {
            return;
        }
        if let Err(error) = shared.session.set_name(Some(&name)).await {
            log::warn!(target: "session_name", "failed to save title: {error}");
        } else {
            let _ = shared
                .events
                .send(SessionEvent::SessionInfoChanged { name: Some(name) });
        }
    });
}

struct GenerationGuard(Weak<SessionShared>);

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        if let Some(shared) = self.0.upgrade() {
            shared.name_generation_active.store(false, Ordering::SeqCst);
        }
    }
}

fn normalize_name(text: &str) -> Option<String> {
    let line = text.lines().find(|line| !line.trim().is_empty())?.trim();
    let line =
        line.trim_matches(|character| matches!(character, '"' | '\'' | '“' | '”' | '`' | '*'));
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let name: String = normalized.chars().take(MAX_NAME_CHARS).collect();
    (!name.trim().is_empty()).then(|| name.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_single_line_and_bounded_in_unicode_characters() {
        assert_eq!(
            normalize_name("\n “天气查询” \n explanation").as_deref(),
            Some("天气查询")
        );
        assert_eq!(
            normalize_name("  Fix   web search  ").as_deref(),
            Some("Fix web search")
        );
        assert_eq!(
            normalize_name(&"中".repeat(100)).unwrap().chars().count(),
            MAX_NAME_CHARS
        );
        assert_eq!(normalize_name("  \n \n"), None);
    }
}

//! Option assembly shared by `stream_simple()` implementations.

use crate::types::{
    Context, Model, SimpleStreamOptions, StreamOptions, ThinkingBudgets, ThinkingLevel,
};

const CONTEXT_SAFETY_TOKENS: u64 = 4096;
const MIN_MAX_TOKENS: u64 = 1;

pub fn clamp_max_tokens_to_context(model: &Model, context: &Context, max_tokens: u64) -> u64 {
    if model.context_window == 0 {
        return max_tokens.max(MIN_MAX_TOKENS);
    }
    let estimate = crate::utils::estimate::estimate_context_tokens(context).tokens;
    let available = model
        .context_window
        .saturating_sub(estimate)
        .saturating_sub(CONTEXT_SAFETY_TOKENS)
        .max(MIN_MAX_TOKENS);
    max_tokens.min(available)
}

/// Merge model defaults and per-request sampling params (request wins per key)
/// into base stream options.
pub fn build_base_options(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
    api_key: Option<&str>,
) -> StreamOptions {
    let sampling_params = match (
        &model.sampling_params,
        options.and_then(|options| options.base.sampling_params.as_ref()),
    ) {
        (Some(model_params), Some(request_params)) => {
            let mut merged = model_params.clone();
            for (key, value) in request_params {
                merged.insert(key.clone(), value.clone());
            }
            Some(merged)
        }
        (model_params, request_params) => model_params
            .as_ref()
            .cloned()
            .or_else(|| request_params.cloned()),
    };
    let mut base = options
        .map(|options| options.base.clone())
        .unwrap_or_default();
    base.temperature = options.and_then(|options| options.base.temperature);
    base.sampling_params = sampling_params;
    base.max_tokens = Some(clamp_max_tokens_to_context(
        model,
        context,
        options
            .and_then(|options| options.base.max_tokens)
            .unwrap_or(model.max_tokens),
    ));
    base.base.api_key = api_key
        .map(str::to_owned)
        .or_else(|| options.and_then(|options| options.base.base.api_key.clone()));
    base
}

/// Tokens always left for the answer when a thinking budget shares the
/// response ceiling.
pub const MIN_ANSWER_TOKENS: u64 = 1024;

pub const DEFAULT_THINKING_BUDGETS: ThinkingBudgets = ThinkingBudgets {
    minimal: Some(1024),
    low: Some(2048),
    medium: Some(8192),
    high: Some(16384),
};

/// Cap `xhigh`/`max` effort down to `high` for token-budget-based providers.
pub fn clamp_reasoning(effort: ThinkingLevel) -> ThinkingLevel {
    match effort {
        ThinkingLevel::Xhigh | ThinkingLevel::Max => ThinkingLevel::High,
        other => other,
    }
}

pub fn thinking_budget_for_level(
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> u64 {
    let budgets = custom_budgets.cloned().unwrap_or_default();
    let level = clamp_reasoning(reasoning_level);
    let resolved = match level {
        ThinkingLevel::Minimal => budgets.minimal.or(DEFAULT_THINKING_BUDGETS.minimal),
        ThinkingLevel::Low => budgets.low.or(DEFAULT_THINKING_BUDGETS.low),
        ThinkingLevel::Medium => budgets.medium.or(DEFAULT_THINKING_BUDGETS.medium),
        ThinkingLevel::High | ThinkingLevel::Xhigh | ThinkingLevel::Max => {
            budgets.high.or(DEFAULT_THINKING_BUDGETS.high)
        }
    };
    resolved.expect("static default budgets are set")
}

/// Cap a thinking budget so at least [`MIN_ANSWER_TOKENS`] remain under a
/// shared response ceiling.
pub fn clamp_thinking_budget_to_answer_room(thinking_budget: u64, ceiling: u64) -> u64 {
    thinking_budget.min(ceiling.saturating_sub(MIN_ANSWER_TOKENS))
}

/// Resolve the response ceiling and thinking budget pair. `None` for
/// `base_max_tokens` means "no explicit caller cap": the model cap applies and
/// thinking fits inside it.
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: Option<u64>,
    model_max_tokens: u64,
    reasoning_level: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> (u64, u64) {
    let mut thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets);
    let max_tokens = match base_max_tokens {
        None => model_max_tokens,
        Some(base) => base.saturating_add(thinking_budget).min(model_max_tokens),
    };

    if max_tokens <= thinking_budget {
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens);
    }

    (max_tokens, thinking_budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, ModelCost, ProviderId};

    fn model(context_window: u64, max_tokens: u64) -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::OpenaiCompletions,
            provider: ProviderId::new("test"),
            base_url: "http://localhost".to_owned(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![crate::types::InputModality::Text],
            cost: ModelCost::default(),
            context_window,
            max_tokens,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    fn context(prompt_chars: usize) -> Context {
        Context {
            system_prompt: Some("x".repeat(prompt_chars)),
            messages: Vec::new(),
            tools: None,
        }
    }

    #[test]
    fn max_tokens_clamped_to_remaining_context() {
        // 4 chars/token: 4000 chars -> 1000 tokens; 4096 safety margin.
        let m = model(8192, 8192);
        let clamped = clamp_max_tokens_to_context(&m, &context(4000), 8192);
        assert_eq!(clamped, 8192 - 1000 - 4096);
    }

    #[test]
    fn zero_context_window_keeps_requested_max() {
        let m = model(0, 512);
        assert_eq!(clamp_max_tokens_to_context(&m, &context(100), 512), 512);
    }

    #[test]
    fn default_budgets_per_level() {
        assert_eq!(
            thinking_budget_for_level(ThinkingLevel::Minimal, None),
            1024
        );
        assert_eq!(thinking_budget_for_level(ThinkingLevel::High, None), 16384);
        assert_eq!(thinking_budget_for_level(ThinkingLevel::Max, None), 16384); // clamped
    }

    #[test]
    fn custom_budgets_override_defaults() {
        let budgets = ThinkingBudgets {
            high: Some(32000),
            ..Default::default()
        };
        assert_eq!(
            thinking_budget_for_level(ThinkingLevel::High, Some(&budgets)),
            32000
        );
    }

    #[test]
    fn thinking_fits_inside_model_ceiling() {
        // The default high budget (16384) exceeds a 4096 ceiling, so it clamps
        // to leave room for the answer.
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(None, 4096, ThinkingLevel::High, None);
        assert_eq!(max_tokens, 4096);
        assert_eq!(budget, 4096 - MIN_ANSWER_TOKENS);
        // Budget exceeding the ceiling leaves room for the answer.
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(None, 2048, ThinkingLevel::High, None);
        assert_eq!(max_tokens, 2048);
        assert_eq!(budget, 2048 - MIN_ANSWER_TOKENS);

        // A ceiling above the default budget keeps the budget unchanged.
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(None, 32768, ThinkingLevel::High, None);
        assert_eq!(max_tokens, 32768);
        assert_eq!(budget, 16384);
    }

    #[test]
    fn explicit_max_tokens_plus_budget_capped_by_model() {
        let (max_tokens, budget) =
            adjust_max_tokens_for_thinking(Some(1000), 4096, ThinkingLevel::Low, None);
        assert_eq!(max_tokens, 1000 + 2048);
        assert_eq!(budget, 2048);
    }
}

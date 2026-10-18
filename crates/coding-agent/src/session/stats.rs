//! Per-session usage statistics.

use std::sync::Mutex;

use serde::Serialize;

use step_harness_providers::types::{Model, Usage};

/// Running totals for one session.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SessionStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// Total cost in the model's currency units (typically USD).
    pub cost: f64,
    /// Number of assistant turns counted.
    pub turns: u64,
}

impl SessionStats {
    /// Total tokens across all buckets.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }
}

/// Accumulates usage across assistant messages.
#[derive(Debug, Default)]
pub struct StatsTracker {
    totals: Mutex<SessionStats>,
}

impl StatsTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one assistant message's usage into the totals.
    pub fn record(&self, model: &Model, usage: &Usage) {
        let mut totals = self
            .totals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut usage = usage.clone();
        let cost = step_harness_providers::models::calculate_cost(model, &mut usage);
        totals.input_tokens += usage.input;
        totals.output_tokens += usage.output;
        totals.cache_read_tokens += usage.cache_read;
        totals.cache_write_tokens += usage.cache_write;
        totals.cost += cost.total;
        totals.turns += 1;
    }

    /// Snapshot of the totals.
    pub fn snapshot(&self) -> SessionStats {
        self.totals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use step_harness_providers::types::{Api, ModelCost, ProviderId};

    fn priced_model() -> Model {
        Model {
            id: "m".to_owned(),
            name: "m".to_owned(),
            api: Api::OPENAI_COMPLETIONS.into(),
            provider: ProviderId::new("p"),
            base_url: "https://example.invalid".to_owned(),
            reasoning: false,
            thinking_level_map: None,
            input: Vec::new(),
            cost: ModelCost {
                rates: step_harness_providers::types::ModelCostRates {
                    input: 3.0,
                    output: 15.0,
                    cache_read: 0.3,
                    cache_write: 3.75,
                },
                tiers: Vec::new(),
            },
            context_window: 1000,
            max_tokens: 100,
            sampling_params: None,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn accumulates_tokens_and_cost() {
        let tracker = StatsTracker::new();
        let usage = Usage {
            input: 1000,
            output: 500,
            cache_read: 0,
            cache_write: 0,
            ..Default::default()
        };
        tracker.record(&priced_model(), &usage);
        tracker.record(&priced_model(), &usage);
        let stats = tracker.snapshot();
        assert_eq!(stats.input_tokens, 2000);
        assert_eq!(stats.output_tokens, 1000);
        assert_eq!(stats.turns, 2);
        // 1000 input * $3/M + 500 output * $15/M = $0.003 + $0.0075 = $0.0105 per turn.
        assert!((stats.cost - 0.021).abs() < 1e-9, "{}", stats.cost);
    }
}

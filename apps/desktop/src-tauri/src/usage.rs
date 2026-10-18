//! SQLite usage query response types. Historical transcripts are not imported.
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageSnapshot {
    pub records: Vec<UsageRecord>,
    pub longest_chat_ms: u64,
    pub unknown_token_requests: u64,
}

#[derive(Serialize)]
pub struct UsageRecord {
    pub timestamp: u64,
    pub provider: String,
    pub model: String,
    pub tokens: u64,
}

//! Redacted provider/runtime diagnostics attached to assistant messages.

use serde::{Deserialize, Serialize};

/// Non-secret error metadata recorded in diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

/// One diagnostic entry on an assistant message.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub kind: String,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Map<String, serde_json::Value>>,
}

impl AssistantMessageDiagnostic {
    pub fn from_http_error(
        kind: impl Into<String>,
        error: &crate::error::ProviderHttpError,
    ) -> Self {
        AssistantMessageDiagnostic {
            kind: kind.into(),
            timestamp: crate::utils::now_ms(),
            error: Some(DiagnosticErrorInfo {
                name: Some("ProviderHttpError".to_owned()),
                message: error.to_string(),
                stack: None,
                code: error.status.map(serde_json::Value::from),
            }),
            details: None,
        }
    }
}

/// Append a diagnostic to a message's diagnostics list.
pub fn append_assistant_message_diagnostic(
    message: &mut crate::types::AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) {
    message
        .diagnostics
        .get_or_insert_with(Vec::new)
        .push(diagnostic);
}

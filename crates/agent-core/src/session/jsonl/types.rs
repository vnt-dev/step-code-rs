//! JSONL session persistence: wire types and decode errors.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\jsonl\types.ts`
//! and `errors.ts`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::harness::types::FileError;
use crate::session::types::{JsonValue, SessionCreateOptions, SessionError, SessionErrorCode};

/// Metadata for a JSONL-backed session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlSessionMetadata {
    pub id: String,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub cwd: String,
    pub path: String,
    /// Filesystem modification time as milliseconds since the Unix epoch.
    pub modified_at: u64,
    pub source_format: u32,
    /// Present only when a v3 parent path could not be resolved to a session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_parent_session_path: Option<String>,
    /// Opaque application-owned metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, JsonValue>>,
}

/// Options for creating a JSONL-backed session.
#[derive(Clone, Debug, Default)]
pub struct JsonlSessionCreateOptions {
    pub id: Option<String>,
    pub parent_session_id: Option<String>,
    pub cwd: String,
    pub metadata: Option<serde_json::Map<String, JsonValue>>,
}

impl From<JsonlSessionCreateOptions> for SessionCreateOptions {
    fn from(options: JsonlSessionCreateOptions) -> Self {
        SessionCreateOptions {
            id: options.id,
            parent_session_id: options.parent_session_id,
        }
    }
}

/// Options for listing JSONL sessions.
#[derive(Clone, Debug, Default)]
pub struct JsonlSessionListOptions {
    pub cwd: Option<String>,
}

/// Options for constructing a [`crate::session::jsonl::JsonlSessionRepo`].
#[derive(Clone, Debug)]
pub struct JsonlSessionRepoOptions {
    /// Root containing coding-agent-compatible cwd-encoded session directories.
    pub sessions_root: String,
}

/// The first line of a v4 session file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonlV4Header {
    pub kind: String,
    pub version: u32,
    pub id: String,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// Preserved only when a v3 parent path could not be resolved to a session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_parent_session_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, JsonValue>>,
}

impl JsonlV4Header {
    /// Build a v4 header for a new session.
    pub fn new(
        id: impl Into<String>,
        created_at: u64,
        cwd: impl Into<String>,
        parent_session_id: Option<String>,
        metadata: Option<serde_json::Map<String, JsonValue>>,
    ) -> Self {
        JsonlV4Header {
            kind: "header".to_owned(),
            version: 4,
            id: id.into(),
            created_at,
            cwd: cwd.into(),
            parent_session_id,
            legacy_parent_session_path: None,
            metadata,
        }
    }
}

/// A line-level decode failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonlDecodeErrorKind {
    /// The line is not valid JSON.
    Syntax,
    /// The line parses but violates the v4 schema.
    Schema,
}

impl JsonlDecodeErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            JsonlDecodeErrorKind::Syntax => "syntax",
            JsonlDecodeErrorKind::Schema => "schema",
        }
    }
}

/// Error returned when a session line cannot be decoded.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct JsonlDecodeError {
    pub kind: JsonlDecodeErrorKind,
    pub message: String,
    #[source]
    pub cause: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl JsonlDecodeError {
    pub fn new(kind: JsonlDecodeErrorKind, message: impl Into<String>) -> Self {
        JsonlDecodeError {
            kind,
            message: message.into(),
            cause: None,
        }
    }

    pub fn syntax(message: impl Into<String>) -> Self {
        JsonlDecodeError::new(JsonlDecodeErrorKind::Syntax, message)
    }

    pub fn schema(message: impl Into<String>) -> Self {
        JsonlDecodeError::new(JsonlDecodeErrorKind::Schema, message)
    }
}

/// Map a filesystem failure onto a session error: not-found stays not-found,
/// everything else is a storage failure.
pub fn file_result<T>(result: Result<T, FileError>, message: &str) -> Result<T, SessionError> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let code = if error.code == crate::harness::types::FileErrorCode::NotFound {
                SessionErrorCode::NotFound
            } else {
                SessionErrorCode::Storage
            };
            Err(SessionError::with_cause(
                code,
                format!("{message}: {}", error.message),
                error,
            ))
        }
    }
}

/// Build the invalid-file error for a failing session line.
pub fn invalid_file(path: &str, line: usize, cause: JsonlDecodeError) -> SessionError {
    SessionError::with_cause(
        SessionErrorCode::InvalidEntry,
        format!(
            "Invalid JSONL v4 session {path}: line {line} {}",
            cause.message
        ),
        cause,
    )
}

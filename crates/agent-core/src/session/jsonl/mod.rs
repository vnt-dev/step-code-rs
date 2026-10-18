//! JSONL v4 session persistence.

pub mod codec;
pub mod repo;
pub mod storage;
pub mod types;

pub use codec::{
    decode_header, decode_mutation, encode_header, encode_mutation, metadata_from_header,
    parse_header, parse_mutation,
};
pub use repo::{
    JsonlSessionRepo, jsonl_session_directory_name, list_jsonl_session_metadata,
    load_jsonl_session_storage, session_file_name,
};
pub use storage::JsonlSessionStorage;
pub use types::{
    JsonlDecodeError, JsonlDecodeErrorKind, JsonlSessionCreateOptions, JsonlSessionListOptions,
    JsonlSessionMetadata, JsonlSessionRepoOptions, JsonlV4Header, file_result, invalid_file,
};

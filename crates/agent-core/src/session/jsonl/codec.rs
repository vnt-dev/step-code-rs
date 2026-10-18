//! JSONL v4 codec: header and mutation line parsing/encoding.
//!
//! Ported from `@step-harness/agent-core` `src\harness\session\jsonl\codec.ts`.

use serde_json::{Map, Value as Json};

use crate::session::jsonl::types::{JsonlDecodeError, JsonlV4Header};
use crate::session::state::SessionMutation;
use crate::session::types::{Entry, LaneRecord};

fn is_object(value: &Json) -> bool {
    matches!(value, Json::Object(_))
}

fn parse_object(line: &str) -> Result<Map<String, Json>, JsonlDecodeError> {
    let value: Json = serde_json::from_str(line)
        .map_err(|error| JsonlDecodeError::syntax(format!("is not valid JSON: {error}")))?;
    if !is_object(&value) {
        return Err(JsonlDecodeError::schema("is not a JSON object"));
    }
    match value {
        Json::Object(map) => Ok(map),
        _ => unreachable!(),
    }
}

fn require_string(value: &Json, field: &str) -> Result<String, JsonlDecodeError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| JsonlDecodeError::schema(format!("has invalid {field}")))
}

fn require_sequence(value: &Json) -> Result<u64, JsonlDecodeError> {
    match value.as_u64() {
        Some(seq) if seq > 0 => Ok(seq),
        _ => Err(JsonlDecodeError::schema("has invalid seq")),
    }
}

fn require_timestamp(value: &Json) -> Result<u64, JsonlDecodeError> {
    value
        .as_u64()
        .ok_or_else(|| JsonlDecodeError::schema("has invalid timestamp"))
}

fn require_nullable_id(value: &Json, field: &str) -> Result<Option<String>, JsonlDecodeError> {
    match value {
        Json::Null => Ok(None),
        Json::String(text) => Ok(Some(text.clone())),
        _ => Err(JsonlDecodeError::schema(format!("has invalid {field}"))),
    }
}

/// Decode the header line.
pub fn decode_header(line: &str) -> Result<JsonlV4Header, JsonlDecodeError> {
    let value = parse_object(line)?;
    if value.get("kind").and_then(Json::as_str) != Some("header") {
        return Err(JsonlDecodeError::schema("is not a header"));
    }
    if value.get("version").and_then(Json::as_u64) != Some(4) {
        return Err(JsonlDecodeError::schema("has unsupported session version"));
    }
    let parent_session_id = match value.get("parentSessionId") {
        None | Some(Json::Null) => None,
        Some(Json::String(text)) => Some(text.clone()),
        Some(_) => return Err(JsonlDecodeError::schema("has invalid parentSessionId")),
    };
    let legacy_parent_session_path = match value.get("legacyParentSessionPath") {
        None | Some(Json::Null) => None,
        Some(Json::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(JsonlDecodeError::schema(
                "has invalid legacyParentSessionPath",
            ));
        }
    };
    if parent_session_id.is_some() && legacy_parent_session_path.is_some() {
        return Err(JsonlDecodeError::schema(
            "has both parentSessionId and legacyParentSessionPath",
        ));
    }
    let metadata = match value.get("metadata") {
        None | Some(Json::Null) => None,
        Some(Json::Object(map)) => Some(map.clone()),
        Some(_) => return Err(JsonlDecodeError::schema("has invalid metadata")),
    };
    Ok(JsonlV4Header {
        kind: "header".to_owned(),
        version: 4,
        id: require_string(&value["id"], "id")?,
        created_at: require_timestamp(&value["createdAt"])?,
        cwd: require_string(&value["cwd"], "cwd")?,
        parent_session_id,
        legacy_parent_session_path,
        metadata,
    })
}

/// Parse a header line.
pub fn parse_header(line: &str) -> Result<JsonlV4Header, JsonlDecodeError> {
    decode_header(line)
}

/// Serialize a header as a JSONL line.
pub fn encode_header(header: &JsonlV4Header) -> String {
    let mut line = serde_json::to_string(header).unwrap_or_else(|_| "{}".to_owned());
    line.push('\n');
    line
}

/// Build session metadata from a header.
pub fn metadata_from_header(
    header: &JsonlV4Header,
    path: &str,
    modified_at: u64,
) -> crate::session::jsonl::types::JsonlSessionMetadata {
    crate::session::jsonl::types::JsonlSessionMetadata {
        id: header.id.clone(),
        created_at: header.created_at,
        parent_session_id: header.parent_session_id.clone(),
        cwd: header.cwd.clone(),
        path: path.to_owned(),
        modified_at,
        source_format: 4,
        legacy_parent_session_path: header.legacy_parent_session_path.clone(),
        metadata: header.metadata.clone(),
    }
}

fn parse_entry_mutation(
    value: &Map<String, Json>,
    seq: u64,
) -> Result<SessionMutation, JsonlDecodeError> {
    let lane = match value.get("lane") {
        None | Some(Json::Null) => None,
        Some(lane) => Some(require_string(lane, "lane")?),
    };
    let entry_type = require_string(&value["type"], "entry type")?;
    if !Entry::ENTRY_TYPES.contains(&entry_type.as_str()) {
        return Err(JsonlDecodeError::schema(format!(
            "has unknown entry type {entry_type}"
        )));
    }
    let parent_id = require_nullable_id(&value["parentId"], "parentId")?;
    let timestamp = require_timestamp(&value["timestamp"])?;
    if entry_type == "custom" {
        require_string(&value["customType"], "customType")?;
    }
    // Build the entry object without the mutation envelope fields.
    let mut entry_value = Json::Object(value.clone());
    entry_value.as_object_mut().unwrap().remove("kind");
    entry_value.as_object_mut().unwrap().remove("lane");
    entry_value["seq"] = Json::from(seq);
    entry_value["timestamp"] = Json::from(timestamp);
    if let Json::Null = &entry_value["parentId"] {
        entry_value
            .as_object_mut()
            .unwrap()
            .insert("parentId".to_owned(), Json::Null);
    } else {
        entry_value["parentId"] = Json::from(parent_id);
    }
    let entry: Entry = serde_json::from_value(entry_value)
        .map_err(|error| JsonlDecodeError::schema(format!("has invalid entry: {error}")))?;
    Ok(match lane {
        Some(lane) => SessionMutation::Entry {
            lane: Some(lane),
            entry,
        },
        None => SessionMutation::Entry { lane: None, entry },
    })
}

fn parse_record_mutation(
    value: &Map<String, Json>,
    seq: u64,
) -> Result<SessionMutation, JsonlDecodeError> {
    let record_type = require_string(&value["type"], "record type")?;
    if !LaneRecord::RECORD_TYPES.contains(&record_type.as_str()) {
        return Err(JsonlDecodeError::schema(format!(
            "has unknown record type {record_type}"
        )));
    }
    let timestamp = require_timestamp(&value["timestamp"])?;
    if record_type == "operation_started" {
        let Some(Json::Object(intent)) = value.get("intent") else {
            return Err(JsonlDecodeError::schema("has invalid intent"));
        };
        let operation_kind = require_string(&intent["kind"], "operation kind")?;
        if !["run", "compaction", "navigation"].contains(&operation_kind.as_str()) {
            return Err(JsonlDecodeError::schema(format!(
                "has unknown operation kind {operation_kind}"
            )));
        }
    }
    if record_type == "operation_finished" {
        require_string(&value["runId"], "runId")?;
    }
    let mut record_value = Json::Object(value.clone());
    record_value.as_object_mut().unwrap().remove("kind");
    record_value["seq"] = Json::from(seq);
    record_value["timestamp"] = Json::from(timestamp);
    let record: LaneRecord = serde_json::from_value(record_value)
        .map_err(|error| JsonlDecodeError::schema(format!("has invalid record: {error}")))?;
    Ok(SessionMutation::Record { record })
}

fn parse_lane_mutation(
    value: &Map<String, Json>,
    seq: u64,
) -> Result<SessionMutation, JsonlDecodeError> {
    Ok(SessionMutation::Lane {
        seq,
        lane: require_string(&value["lane"], "lane")?,
        leaf_id: require_nullable_id(&value["leafId"], "leafId")?,
    })
}

fn parse_fact_mutation(
    value: &Map<String, Json>,
    seq: u64,
) -> Result<SessionMutation, JsonlDecodeError> {
    match value.get("fact").and_then(Json::as_str) {
        Some("name") => {
            let name = match value.get("name") {
                None | Some(Json::Null) => None,
                Some(Json::String(text)) => Some(text.clone()),
                Some(_) => return Err(JsonlDecodeError::schema("has invalid name")),
            };
            Ok(SessionMutation::name(seq, name))
        }
        Some("label") => {
            let label = match value.get("label") {
                None | Some(Json::Null) => None,
                Some(Json::String(text)) => Some(text.clone()),
                Some(_) => return Err(JsonlDecodeError::schema("has invalid label")),
            };
            Ok(SessionMutation::label(
                seq,
                require_string(&value["targetId"], "targetId")?,
                label,
            ))
        }
        Some("project") => {
            let project = match value.get("project") {
                None | Some(Json::Null) => None,
                Some(Json::String(text)) => Some(text.clone()),
                Some(_) => return Err(JsonlDecodeError::schema("has invalid project")),
            };
            Ok(SessionMutation::project(seq, project))
        }
        _ => Err(JsonlDecodeError::schema("has unknown fact type")),
    }
}

/// Decode one mutation line.
pub fn decode_mutation(line: &str) -> Result<SessionMutation, JsonlDecodeError> {
    let value = parse_object(line)?;
    let seq = require_sequence(&value["seq"])?;
    match value.get("kind").and_then(Json::as_str) {
        Some("entry") => parse_entry_mutation(&value, seq),
        Some("record") => parse_record_mutation(&value, seq),
        Some("lane") => parse_lane_mutation(&value, seq),
        Some("fact") => parse_fact_mutation(&value, seq),
        _ => Err(JsonlDecodeError::schema("has unknown mutation kind")),
    }
}

/// Parse one mutation line.
pub fn parse_mutation(line: &str) -> Result<SessionMutation, JsonlDecodeError> {
    decode_mutation(line)
}

/// Serialize a mutation as a JSONL line.
pub fn encode_mutation(mutation: &SessionMutation) -> String {
    let line = match mutation {
        SessionMutation::Entry { lane, entry } => {
            let mut object = Map::new();
            object.insert("kind".to_owned(), Json::from("entry"));
            if let Some(lane) = lane {
                object.insert("lane".to_owned(), Json::from(lane.clone()));
            }
            if let Json::Object(entry_map) = serde_json::to_value(entry).unwrap_or(Json::Null) {
                object.extend(entry_map);
            }
            Json::Object(object)
        }
        SessionMutation::Record { record } => {
            let mut object = Map::new();
            object.insert("kind".to_owned(), Json::from("record"));
            if let Json::Object(record_map) = serde_json::to_value(record).unwrap_or(Json::Null) {
                object.extend(record_map);
            }
            Json::Object(object)
        }
        SessionMutation::Lane { .. } | SessionMutation::Fact { .. } => {
            serde_json::to_value(mutation).unwrap_or(Json::Null)
        }
    };
    let mut encoded = serde_json::to_string(&line).unwrap_or_else(|_| "{}".to_owned());
    encoded.push('\n');
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::jsonl::types::JsonlDecodeErrorKind;
    use crate::session::types::ProvisionedEntry;
    use crate::types::AgentMessage;

    #[test]
    fn header_round_trip() {
        let header = JsonlV4Header::new("s1", 1759999999999, "/work", None, None);
        let encoded = encode_header(&header);
        assert!(encoded.ends_with('\n'));
        let decoded = parse_header(encoded.trim_end()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn rejects_non_v4_headers() {
        let line = r#"{"kind":"header","version":3,"id":"s","createdAt":1,"cwd":"/w"}"#;
        let error = parse_header(line).unwrap_err();
        assert_eq!(error.kind, JsonlDecodeErrorKind::Schema);
        let line = r#"{"kind":"other","version":4,"id":"s","createdAt":1,"cwd":"/w"}"#;
        assert_eq!(
            parse_header(line).unwrap_err().kind,
            JsonlDecodeErrorKind::Schema
        );
    }

    #[test]
    fn entry_mutation_round_trip() {
        let mutation = SessionMutation::Entry {
            lane: Some("main".to_owned()),
            entry: ProvisionedEntry::Message {
                id: "e1".to_owned(),
                message: AgentMessage::user("hello", 42),
                terminate: false,
            }
            .into_entry(None, 1, 1000),
        };
        let encoded = encode_mutation(&mutation);
        let decoded = parse_mutation(encoded.trim_end()).unwrap();
        assert_eq!(decoded, mutation);
        assert!(encoded.starts_with("{\"kind\":\"entry\",\"lane\":\"main\""));
    }

    #[test]
    fn record_mutation_round_trip() {
        let mutation = SessionMutation::Record {
            record: LaneRecord::AbortRequested(crate::session::types::AbortRequestedRecord {
                id: "r1".to_owned(),
                seq: 2,
                lane: "main".to_owned(),
                timestamp: 2000,
                run_id: "run-1".to_owned(),
            }),
        };
        let encoded = encode_mutation(&mutation);
        let decoded = parse_mutation(encoded.trim_end()).unwrap();
        assert_eq!(decoded, mutation);
    }

    #[test]
    fn lane_and_fact_mutations_round_trip() {
        let lane = SessionMutation::Lane {
            seq: 1,
            lane: "alt".to_owned(),
            leaf_id: Some("e1".to_owned()),
        };
        assert_eq!(
            parse_mutation(encode_mutation(&lane).trim_end()).unwrap(),
            lane
        );

        let fact = SessionMutation::name(2, Some("named".to_owned()));
        assert_eq!(
            parse_mutation(encode_mutation(&fact).trim_end()).unwrap(),
            fact
        );

        let label = SessionMutation::label(3, "e1".to_owned(), None);
        assert_eq!(
            parse_mutation(encode_mutation(&label).trim_end()).unwrap(),
            label
        );

        let project = SessionMutation::project(4, Some("step-code".to_owned()));
        assert_eq!(
            parse_mutation(encode_mutation(&project).trim_end()).unwrap(),
            project
        );
        let cleared = SessionMutation::project(5, None);
        assert_eq!(
            parse_mutation(encode_mutation(&cleared).trim_end()).unwrap(),
            cleared
        );
    }

    #[test]
    fn rejects_invalid_lines() {
        assert_eq!(
            parse_mutation("not json").unwrap_err().kind,
            JsonlDecodeErrorKind::Syntax
        );
        assert_eq!(
            parse_mutation(r#"{"kind":"entry","seq":0,"id":"e","type":"message","parentId":null,"timestamp":1}"#)
                .unwrap_err()
                .kind,
            JsonlDecodeErrorKind::Schema
        );
        assert_eq!(
            parse_mutation(
                r#"{"kind":"entry","seq":1,"id":"e","type":"nope","parentId":null,"timestamp":1}"#
            )
            .unwrap_err()
            .kind,
            JsonlDecodeErrorKind::Schema
        );
        assert_eq!(
            parse_mutation(r#"{"kind":"nope","seq":1}"#)
                .unwrap_err()
                .kind,
            JsonlDecodeErrorKind::Schema
        );
    }
}

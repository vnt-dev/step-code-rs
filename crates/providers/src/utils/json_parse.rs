//! JSON repair and streaming-tolerant parsing for provider payloads.
//!
//! - [`repair_json`] escapes raw control characters and invalid escapes inside
//!   string literals, mirroring the TypeScript implementation.
//! - [`parse_streaming_json`] additionally completes truncated JSON from
//!   streaming tool-call arguments (the `partial-json` dependency's role).

use serde_json::Value;

/// Repairs malformed JSON string literals by:
/// - escaping raw control characters inside strings
/// - doubling backslashes before invalid escape characters
pub fn repair_json(json: &str) -> String {
    const VALID_JSON_ESCAPES: [char; 9] = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'];

    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut chars = json.chars().peekable();

    while let Some(char) = chars.next() {
        if !in_string {
            repaired.push(char);
            if char == '"' {
                in_string = true;
            }
            continue;
        }

        if char == '"' {
            repaired.push(char);
            in_string = false;
            continue;
        }

        if char == '\\' {
            let Some(next) = chars.peek().copied() else {
                repaired.push_str("\\\\");
                continue;
            };

            if next == 'u' {
                let mut unicode_digits = String::new();
                {
                    let mut lookahead = chars.clone();
                    lookahead.next(); // Skip the 'u' before reading four hex digits.
                    for _ in 0..4 {
                        match lookahead.next() {
                            Some(digit @ ('0'..='9' | 'a'..='f' | 'A'..='F')) => {
                                unicode_digits.push(digit)
                            }
                            _ => break,
                        }
                    }
                }
                if unicode_digits.len() == 4 {
                    repaired.push('\\');
                    repaired.push('u');
                    repaired.push_str(&unicode_digits);
                    for _ in 0..5 {
                        chars.next();
                    }
                    continue;
                }
            }

            if VALID_JSON_ESCAPES.contains(&next) {
                repaired.push('\\');
                repaired.push(next);
                chars.next();
                continue;
            }

            repaired.push_str("\\\\");
            continue;
        }

        if char.is_control() {
            match char {
                '\u{8}' => repaired.push_str("\\b"),
                '\u{c}' => repaired.push_str("\\f"),
                '\n' => repaired.push_str("\\n"),
                '\r' => repaired.push_str("\\r"),
                '\t' => repaired.push_str("\\t"),
                other => repaired.push_str(&format!("\\u{:04x}", other as u32)),
            }
        } else {
            repaired.push(char);
        }
    }

    repaired
}

/// Parse JSON, falling back to the repaired form when the original fails.
pub fn parse_json_with_repair(json: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str(json) {
        Ok(value) => Ok(value),
        Err(original_error) => {
            let repaired = repair_json(json);
            if repaired != json {
                serde_json::from_str(&repaired).map_err(|_| original_error)
            } else {
                Err(original_error)
            }
        }
    }
}

/// Parse an in-flight object without inventing property values. Truncated
/// strings are retained; incomplete literals preserve earlier complete fields.
pub fn parse_streaming_json(partial_json: Option<&str>) -> Value {
    let Some(partial) = partial_json.filter(|text| !text.trim().is_empty()) else {
        return Value::Object(Default::default());
    };
    if let Ok(value) = parse_json_with_repair(partial) {
        return value;
    }
    for candidate in [partial.to_owned(), repair_json(partial)] {
        if let Ok(value) = jiter::JsonValue::parse_with_config(
            candidate.as_bytes(),
            false,
            jiter::PartialMode::TrailingStrings,
        ) && let Some(value) = from_partial_value(&value)
        {
            return value;
        }
    }
    Value::Object(Default::default())
}

fn from_partial_value(value: &jiter::JsonValue<'_>) -> Option<Value> {
    use jiter::JsonValue;
    Some(match value {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(value) => Value::Bool(*value),
        JsonValue::Int(value) => Value::from(*value),
        JsonValue::Float(value) => Value::Number(serde_json::Number::from_f64(*value)?),
        JsonValue::Str(value) => Value::String(value.to_string()),
        JsonValue::Array(items) => Value::Array(
            items
                .iter()
                .map(from_partial_value)
                .collect::<Option<_>>()?,
        ),
        JsonValue::Object(entries) => Value::Object(
            entries
                .iter()
                .map(|(key, value)| Some((key.to_string(), from_partial_value(value)?)))
                .collect::<Option<_>>()?,
        ),
    })
}

/// Final tool arguments must be a complete, valid JSON object. Tolerant
/// streaming previews must never determine executable arguments.
pub fn parse_tool_arguments(
    json: &str,
) -> Result<serde_json::Map<String, Value>, ToolArgumentsError> {
    let value: Value = serde_json::from_str(json)?;
    match value {
        Value::Object(arguments) => Ok(arguments),
        _ => Err(ToolArgumentsError::NotObject),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolArgumentsError {
    #[error("Invalid tool argument JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Tool arguments must be a JSON object")]
    NotObject,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_raw_control_characters() {
        let repaired = repair_json("{\"a\": \"line1\nline2\ttab\"}");
        assert_eq!(repaired, "{\"a\": \"line1\\nline2\\ttab\"}");
        let parsed: Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["a"], "line1\nline2\ttab");
    }

    #[test]
    fn repairs_invalid_escapes() {
        // \t is a valid escape and stays verbatim; \x is invalid and doubled.
        let repaired = repair_json(r#"{"path": "C:\temp\x"}"#);
        assert_eq!(repaired, r#"{"path": "C:\temp\\x"}"#);
        assert!(serde_json::from_str::<Value>(&repaired).is_ok());
    }

    #[test]
    fn preserves_valid_escapes_and_unicode() {
        let input = r#"{"a": "é\n😀"}"#;
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn parse_with_repair_prefers_original() {
        let value = parse_json_with_repair("{\"a\": 1}").unwrap();
        assert_eq!(value["a"], 1);
    }

    #[test]
    fn streaming_parse_completes_partial_objects() {
        let value = parse_streaming_json(Some(r#"{"path": "/sr"#));
        assert_eq!(value["path"], "/sr");

        let value = parse_streaming_json(Some(r#"{"path": "/src", "line": 12, "#));
        assert_eq!(value["path"], "/src");
        assert_eq!(value["line"], 12);
        assert_eq!(value["null"], Value::Null);
    }

    #[test]
    fn streaming_parse_complete_json_wins() {
        let value = parse_streaming_json(Some(r#"{"a": [1, 2, 3]}"#));
        assert_eq!(value["a"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn streaming_parse_empty_returns_empty_object() {
        assert_eq!(
            parse_streaming_json(None),
            Value::Object(Default::default())
        );
        assert_eq!(
            parse_streaming_json(Some("")),
            Value::Object(Default::default())
        );
    }

    #[test]
    fn streaming_parse_nested_arrays() {
        let value = parse_streaming_json(Some(r#"{"items": [{"a": 1}, {"b": "#));
        let items = value["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["a"], 1);
    }

    #[test]
    fn preserves_completed_fields_before_incomplete_values() {
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":1,"b":"x""#)),
            serde_json::json!({"a":1,"b":"x"})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":1,"b":tru"#)),
            serde_json::json!({"a":1})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":1,"b":1e+"#)),
            serde_json::json!({"a":1})
        );
        assert_eq!(
            parse_streaming_json(Some(r#"{"a":"中文😀","b":fal"#)),
            serde_json::json!({"a":"中文😀"})
        );
    }

    #[test]
    fn final_arguments_require_a_valid_complete_object() {
        for input in [r#"{"a":1,"b":tru"#, r#"{"a":"bad\x"}"#, "null", "[]"] {
            assert!(parse_tool_arguments(input).is_err(), "accepted {input}");
        }
        assert_eq!(parse_tool_arguments(r#"{"a":1}"#).unwrap()["a"], 1);
    }

    #[test]
    fn unicode_escape_repair_keeps_valid_escapes() {
        assert_eq!(repair_json(r#"{"a":"\u4e2d\x"}"#), r#"{"a":"\u4e2d\\x"}"#);
    }
}

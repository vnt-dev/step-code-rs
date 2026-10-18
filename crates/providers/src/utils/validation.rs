//! Tool argument validation and coercion against JSON Schemas, backed by the
//! `jsonschema` crate (the TypeScript source uses typebox's compiler).

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use jsonschema::Validator;
use jsonschema::error::ValidationErrorKind;
use serde_json::Value;

use crate::types::{Tool, ToolCall};

/// Bounded cache; compilation is coalesced per canonical schema, not under a global lock.
static VALIDATOR_CACHE: LazyLock<moka::sync::Cache<String, Arc<Validator>>> =
    LazyLock::new(|| moka::sync::Cache::new(256));

fn validator_for(schema: &Value) -> Result<Arc<Validator>, ToolValidationError> {
    let mut canonical = schema.clone();
    canonical.sort_all_objects();
    VALIDATOR_CACHE
        .try_get_with(canonical.to_string(), || {
            jsonschema::validator_for(schema)
                .map(Arc::new)
                .map_err(|error| ToolValidationError {
                    kind: ToolValidationErrorKind::InvalidSchema,
                    message: format!("Invalid tool JSON Schema: {error}"),
                })
        })
        .map_err(|error| ToolValidationError {
            kind: error.kind,
            message: error.message.clone(),
        })
}

/// A validation failure with a rendered, tool-call-friendly message.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ToolValidationError {
    pub kind: ToolValidationErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolValidationErrorKind {
    InvalidSchema,
    UnknownTool,
    InvalidArguments,
}

fn schema_types(schema: &Value) -> Vec<String> {
    match schema.get("type") {
        Some(Value::String(single)) => vec![single.clone()],
        Some(Value::Array(many)) => many
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, json_type: &str) -> bool {
    match json_type {
        "number" => value.is_number(),
        "integer" => {
            value.as_u64().is_some() || value.as_i64().is_some() || {
                value.as_f64().is_some_and(|float| float.fract() == 0.0)
            }
        }
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn coerce_primitive_by_type(value: &Value, json_type: &str) -> Value {
    match json_type {
        "number" => match value {
            Value::Null => Value::from(0),
            Value::String(text) if !text.trim().is_empty() => text
                .trim()
                .parse::<f64>()
                .map(|parsed| {
                    if parsed.fract() == 0.0 && parsed.abs() <= i64::MAX as f64 {
                        // Integral floats serialize as integers (JS parity).
                        Value::from(parsed as i64)
                    } else {
                        Value::from(parsed)
                    }
                })
                .ok()
                .unwrap_or_else(|| value.clone()),
            Value::Bool(flag) => Value::from(if *flag { 1 } else { 0 }),
            _ => value.clone(),
        },
        "integer" => match value {
            Value::Null => Value::from(0),
            Value::String(text) if !text.trim().is_empty() => text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|float| float.fract() == 0.0)
                .map(|float| Value::from(float as i64))
                .unwrap_or_else(|| value.clone()),
            Value::Bool(flag) => Value::from(if *flag { 1 } else { 0 }),
            _ => value.clone(),
        },
        "boolean" => match value {
            Value::Null => Value::from(false),
            Value::String(text) if text == "true" => Value::from(true),
            Value::String(text) if text == "false" => Value::from(false),
            Value::Number(number) if number.as_i64() == Some(1) => Value::from(true),
            Value::Number(number) if number.as_i64() == Some(0) => Value::from(false),
            _ => value.clone(),
        },
        "string" => match value {
            Value::Null => Value::String(String::new()),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Bool(flag) => Value::String(flag.to_string()),
            _ => value.clone(),
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Value::Null,
            Value::Number(number) if number.as_i64() == Some(0) => Value::Null,
            Value::Bool(flag) if !*flag => Value::Null,
            _ => value.clone(),
        },
        _ => value.clone(),
    }
}

fn coerce_with_json_schema(value: &Value, schema: &Value) -> Value {
    let mut next = value.clone();

    for nested in schema
        .get("allOf")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        next = coerce_with_json_schema(&next, nested);
    }

    for union_key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(union_key).and_then(Value::as_array) {
            next = coerce_with_union_schema(&next, variants);
        }
    }

    let types = schema_types(schema);
    let matches_union_member = types.len() > 1
        && types
            .iter()
            .any(|schema_type| matches_json_type(&next, schema_type));
    if !types.is_empty() && !matches_union_member {
        for schema_type in &types {
            let candidate = coerce_primitive_by_type(&next, schema_type);
            if candidate != next {
                next = candidate;
                break;
            }
        }
    }

    if types.iter().any(|t| t == "object")
        && let Some(map) = next.as_object()
    {
        let mut coerced = serde_json::Map::new();
        let properties = schema.get("properties").and_then(Value::as_object);
        for (key, property_value) in map {
            let property_schema = properties.and_then(|props| props.get(key));
            let coerced_value = match property_schema {
                Some(property_schema) => coerce_with_json_schema(property_value, property_schema),
                None => property_value.clone(),
            };
            coerced.insert(key.clone(), coerced_value);
        }
        // additionalProperties with a schema applies to keys not in properties.
        if let Some(additional) = schema
            .get("additionalProperties")
            .filter(|value| value.is_object())
            && let Some(properties) = properties
        {
            let additional = additional.clone();
            for (key, property_value) in coerced.iter_mut() {
                if !properties.contains_key(key) {
                    *property_value = coerce_with_json_schema(property_value, &additional);
                }
            }
        }
        next = Value::Object(coerced);
    }

    if types.iter().any(|t| t == "array")
        && let Some(items) = next.as_array()
    {
        let mut coerced_items = Vec::with_capacity(items.len());
        match schema.get("items") {
            Some(Value::Array(tuple_schemas)) => {
                for (index, item) in items.iter().enumerate() {
                    let item_schema = tuple_schemas.get(index);
                    coerced_items.push(match item_schema {
                        Some(item_schema) => coerce_with_json_schema(item, item_schema),
                        None => item.clone(),
                    });
                }
            }
            Some(item_schema @ Value::Object(_)) => {
                for item in items {
                    coerced_items.push(coerce_with_json_schema(item, item_schema));
                }
            }
            _ => coerced_items.extend(items.iter().cloned()),
        }
        next = Value::Array(coerced_items);
    }

    next
}

fn coerce_with_union_schema(value: &Value, variants: &[Value]) -> Value {
    for schema in variants {
        if let Ok(validator) = validator_for(schema)
            && validator.is_valid(value)
        {
            return value.clone();
        }
    }
    for schema in variants {
        let coerced = coerce_with_json_schema(value, schema);
        if let Ok(validator) = validator_for(schema)
            && validator.is_valid(&coerced)
        {
            return coerced;
        }
    }
    value.clone()
}

/// Delete `null` values for optional properties whose schema rejects null,
/// recursively (the typebox `normalizeOptionalNulls` port).
fn normalize_optional_nulls(value: &mut Value, schema: &Value) {
    let Some(items) = value.as_array_mut() else {
        if let Some(map) = value.as_object_mut() {
            normalize_object_nulls(map, schema);
        }
        return;
    };
    match schema.get("items") {
        Some(Value::Array(tuple_schemas)) => {
            for (index, item) in items.iter_mut().enumerate() {
                if let Some(item_schema) = tuple_schemas.get(index) {
                    normalize_optional_nulls(item, item_schema);
                }
            }
        }
        Some(item_schema @ Value::Object(_)) => {
            for item in items {
                normalize_optional_nulls(item, item_schema);
            }
        }
        _ => {}
    }
}

fn normalize_object_nulls(map: &mut serde_json::Map<String, Value>, schema: &Value) {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let required: HashSet<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|entries| entries.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    let keys: Vec<String> = map.keys().cloned().collect();
    for key in keys {
        let Some(property_schema) = properties.get(&key) else {
            continue;
        };
        let is_null = map.get(&key).is_some_and(Value::is_null);
        if is_null && !required.contains(key.as_str()) {
            let null_rejected = validator_for(property_schema)
                .is_ok_and(|validator| !validator.is_valid(&Value::Null));
            if null_rejected {
                map.remove(&key);
                continue;
            }
        }
        if let Some(property_value) = map.get_mut(&key) {
            normalize_optional_nulls(property_value, property_schema);
        }
    }
}

/// Format a validation error path like the TypeScript implementation:
/// `required` errors name the missing property, others use dotted instance paths.
fn format_validation_path(error: &jsonschema::error::ValidationError<'_>) -> String {
    let instance_path = error.instance_path().as_str();
    let path = instance_path.trim_start_matches('/').replace('/', ".");
    if let ValidationErrorKind::Required { property } = error.kind() {
        let property = property.as_str().unwrap_or_default();
        return if path.is_empty() {
            property.to_owned()
        } else {
            format!("{path}.{property}")
        };
    }
    if path.is_empty() {
        "root".to_owned()
    } else {
        path
    }
}

/// Find a tool by name and validate the tool call arguments against its schema.
pub fn validate_tool_call(
    tools: &[Tool],
    tool_call: &ToolCall,
) -> Result<serde_json::Map<String, Value>, ToolValidationError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == tool_call.name)
        .ok_or_else(|| ToolValidationError {
            kind: ToolValidationErrorKind::UnknownTool,
            message: format!("Tool \"{}\" not found", tool_call.name),
        })?;
    validate_tool_arguments(tool, tool_call)
}

/// Validate (and coerce) tool call arguments against the tool's JSON Schema.
///
/// Mirrors the TypeScript pipeline:
/// 1. drop `null` for optional properties that reject null,
/// 2. coerce stringly-typed values into their declared JSON types,
/// 3. validate, rendering a per-path error list on failure.
pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &ToolCall,
) -> Result<serde_json::Map<String, Value>, ToolValidationError> {
    let validator = validator_for(&tool.parameters)?;
    let mut args_value = Value::Object(tool_call.arguments.clone());

    normalize_optional_nulls(&mut args_value, &tool.parameters);
    let coerced = coerce_with_json_schema(&args_value, &tool.parameters);

    let candidate = coerced.as_object().cloned().unwrap_or_default();
    if validator.is_valid(&coerced) {
        return Ok(candidate);
    }

    let mut error_lines = Vec::new();
    for error in validator.iter_errors(&coerced) {
        error_lines.push(format!("  - {}: {}", format_validation_path(&error), error));
    }
    let errors = if error_lines.is_empty() {
        "Unknown validation error".to_owned()
    } else {
        error_lines.join("\n")
    };

    let received = serde_json::to_string_pretty(&tool_call.arguments).unwrap_or_default();
    Err(ToolValidationError {
        kind: ToolValidationErrorKind::InvalidArguments,
        message: format!(
            "Validation failed for tool \"{}\":\n{errors}\n\nReceived arguments:\n{received}",
            tool_call.name
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConstrainedSamplingConfig;

    #[test]
    fn invalid_schema_is_an_error_rather_than_a_validation_bypass() {
        let error = validate_tool_arguments(
            &tool(serde_json::json!({"type":"invalid_type"})),
            &call(serde_json::json!({})),
        )
        .unwrap_err();
        assert_eq!(error.kind, ToolValidationErrorKind::InvalidSchema);
    }

    #[test]
    fn reordered_schema_properties_share_a_compiled_validator() {
        let first: Value = serde_json::from_str(
            r#"{"type":"object","properties":{"cache_unique_test":{"type":"string"}}}"#,
        )
        .unwrap();
        let second: Value = serde_json::from_str(
            r#"{"properties":{"cache_unique_test":{"type":"string"}},"type":"object"}"#,
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &validator_for(&first).unwrap(),
            &validator_for(&second).unwrap()
        ));
    }

    fn tool(parameters: Value) -> Tool {
        Tool {
            name: "test".to_owned(),
            description: "test tool".to_owned(),
            parameters,
            constrained_sampling: None::<ConstrainedSamplingConfig>,
        }
    }

    fn call(arguments: Value) -> ToolCall {
        ToolCall {
            id: "1".to_owned(),
            name: "test".to_owned(),
            arguments: arguments.as_object().cloned().unwrap(),
            thought_signature: None,
            namespace: None,
        }
    }

    #[test]
    fn validates_valid_arguments() {
        let tool = tool(serde_json::json!({
            "type": "object",
            "properties": { "count": { "type": "number" }, "label": { "type": "string" } },
            "required": ["count"]
        }));
        let validated =
            validate_tool_arguments(&tool, &call(serde_json::json!({"count": 3, "label": "x"})))
                .unwrap();
        assert_eq!(validated["count"], 3);
    }

    #[test]
    fn coerces_string_numbers() {
        let tool = tool(serde_json::json!({
            "type": "object",
            "properties": { "count": { "type": "number" }, "flag": { "type": "boolean" } },
            "required": ["count", "flag"]
        }));
        let validated = validate_tool_arguments(
            &tool,
            &call(serde_json::json!({"count": "5", "flag": "true"})),
        )
        .unwrap();
        assert_eq!(validated["count"], 5);
        assert_eq!(validated["flag"], true);
    }

    #[test]
    fn missing_required_reports_property() {
        let tool = tool(serde_json::json!({
            "type": "object",
            "properties": { "nested": { "type": "object", "properties": { "inner": { "type": "string" } }, "required": ["inner"] } },
            "required": ["nested"]
        }));
        let error =
            validate_tool_arguments(&tool, &call(serde_json::json!({"nested": {}}))).unwrap_err();
        assert!(
            error.message.contains("nested.inner"),
            "message was: {}",
            error.message
        );
    }

    #[test]
    fn optional_nulls_removed() {
        let tool = tool(serde_json::json!({
            "type": "object",
            "properties": { "a": { "type": "string" } },
            "required": []
        }));
        let validated =
            validate_tool_arguments(&tool, &call(serde_json::json!({"a": null}))).unwrap();
        assert!(validated.get("a").is_none());
    }

    #[test]
    fn unknown_tool_errors() {
        let error = validate_tool_call(
            &[tool(serde_json::json!({"type": "object"}))],
            &ToolCall {
                id: "1".to_owned(),
                name: "other".to_owned(),
                arguments: Default::default(),
                thought_signature: None,
                namespace: None,
            },
        )
        .unwrap_err();
        assert!(error.message.contains("not found"));
    }

    #[test]
    fn union_type_coercion() {
        let tool = tool(serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": ["string", "number"] } },
            "required": ["value"]
        }));
        let validated =
            validate_tool_arguments(&tool, &call(serde_json::json!({"value": 42}))).unwrap();
        assert_eq!(validated["value"], 42);
    }
}

//! Provider-side constrained sampling: strict JSON Schema conversion and
//! grammar tool handling.

use std::collections::HashMap;

use crate::types::{GrammarFormat, StrictMode, Tool};

/// Error converting a schema to the strict subset providers accept.
#[derive(Debug, thiserror::Error)]
#[error("unsupported strict JSON Schema: {0}")]
pub struct UnsupportedStrictJsonSchema(pub String);

const UNSUPPORTED_STRICT_SCHEMA_KEYS: &[&str] = &[
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

fn is_json_schema_object(value: &serde_json::Value) -> bool {
    value.is_object()
}

fn schema_types(schema: &serde_json::Value) -> Vec<String> {
    match schema.get("type") {
        Some(serde_json::Value::String(single)) => vec![single.clone()],
        Some(serde_json::Value::Array(many)) => many
            .iter()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn is_structured_schema(schema: &serde_json::Value) -> bool {
    if !is_json_schema_object(schema) {
        return false;
    }
    let types = schema_types(schema);
    types.iter().any(|t| t == "object" || t == "array")
        || schema.get("properties").is_some()
        || schema.get("items").is_some()
}

fn schema_allows_null(schema: &serde_json::Value) -> bool {
    if !is_json_schema_object(schema) {
        return false;
    }
    if schema.get("type") == Some(&serde_json::Value::String("null".to_owned())) {
        return true;
    }
    if schema
        .get("type")
        .and_then(|value| value.as_array())
        .is_some_and(|types| types.contains(&serde_json::Value::String("null".to_owned())))
    {
        return true;
    }
    if schema.get("const") == Some(&serde_json::Value::Null) {
        return true;
    }
    if let Some(variants) = schema.get("enum").and_then(|value| value.as_array())
        && variants.contains(&serde_json::Value::Null)
    {
        return true;
    }
    schema
        .get("anyOf")
        .and_then(|value| value.as_array())
        .is_some_and(|variants| variants.iter().any(schema_allows_null))
}

fn make_json_schema_node_strict(
    schema: &mut serde_json::Value,
) -> Result<(), UnsupportedStrictJsonSchema> {
    if !is_json_schema_object(schema) {
        return Err(UnsupportedStrictJsonSchema(
            "boolean schemas are unsupported".to_owned(),
        ));
    }
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if schema.get(key).is_some() {
            return Err(UnsupportedStrictJsonSchema(format!(
                "{key} schemas are unsupported"
            )));
        }
    }

    if let Some(any_of) = schema.get_mut("anyOf") {
        let variants = any_of.as_array_mut().ok_or_else(|| {
            UnsupportedStrictJsonSchema("anyOf must contain at least one schema".to_owned())
        })?;
        if variants.is_empty() {
            return Err(UnsupportedStrictJsonSchema(
                "anyOf must contain at least one schema".to_owned(),
            ));
        }
        for variant in variants {
            if is_structured_schema(variant) {
                return Err(UnsupportedStrictJsonSchema(
                    "object and array unions are unsupported".to_owned(),
                ));
            }
            make_json_schema_node_strict(variant)?;
        }
    }

    if let Some(items) = schema.get_mut("items") {
        if items.is_array() {
            return Err(UnsupportedStrictJsonSchema(
                "tuple schemas are unsupported".to_owned(),
            ));
        }
        make_json_schema_node_strict(items)?;
    }

    let is_object_schema =
        schema.get("type") == Some(&serde_json::Value::String("object".to_owned()));
    if schema.get("properties").is_some() && !is_object_schema {
        return Err(UnsupportedStrictJsonSchema(
            "properties require type object".to_owned(),
        ));
    }
    if !is_object_schema {
        return Ok(());
    }
    match schema.get("additionalProperties") {
        Some(serde_json::Value::Bool(false)) => {}
        Some(_) => {
            return Err(UnsupportedStrictJsonSchema(
                "schema-valued or true additionalProperties is unsupported".to_owned(),
            ));
        }
        None => {}
    }
    if let Some(required) = schema.get("required") {
        let valid = required
            .as_array()
            .is_some_and(|entries| entries.iter().all(|entry| entry.is_string()));
        if !valid {
            return Err(UnsupportedStrictJsonSchema(
                "object required must be a string array".to_owned(),
            ));
        }
    }

    let properties = schema
        .get("properties")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let property_names: Vec<String> = properties.keys().cloned().collect();
    let required_names: std::collections::HashSet<String> = schema
        .get("required")
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    if required_names
        .iter()
        .any(|name| !property_names.contains(name))
    {
        return Err(UnsupportedStrictJsonSchema(
            "required contains an unknown property".to_owned(),
        ));
    }
    let mut strict_properties = serde_json::Map::new();
    for (key, mut property) in properties {
        make_json_schema_node_strict(&mut property)?;
        if !required_names.contains(&key) && !schema_allows_null(&property) {
            strict_properties.insert(
                key,
                serde_json::json!({ "anyOf": [property, { "type": "null" }] }),
            );
        } else {
            strict_properties.insert(key, property);
        }
    }
    schema.as_object_mut().unwrap().insert(
        "required".to_owned(),
        serde_json::Value::Array(
            property_names
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    schema.as_object_mut().unwrap().insert(
        "additionalProperties".to_owned(),
        serde_json::Value::Bool(false),
    );
    schema.as_object_mut().unwrap().insert(
        "properties".to_owned(),
        serde_json::Value::Object(strict_properties),
    );
    Ok(())
}

/// Convert a tool schema to the strict subset expected by provider
/// constrained sampling: optional properties become `anyOf` with null, all
/// properties are required, and `additionalProperties` is false.
pub fn make_strict_json_schema(
    schema: &serde_json::Value,
) -> Result<serde_json::Value, UnsupportedStrictJsonSchema> {
    let mut cloned = schema.clone();
    if !is_json_schema_object(&cloned) {
        return Err(UnsupportedStrictJsonSchema(
            "root schema must have type object".to_owned(),
        ));
    }
    make_json_schema_node_strict(&mut cloned)?;
    if cloned.get("type") != Some(&serde_json::Value::String("object".to_owned())) {
        return Err(UnsupportedStrictJsonSchema(
            "root schema must have type object".to_owned(),
        ));
    }
    Ok(cloned)
}

/// The tool parameters to send, strict-converted when requested.
pub fn get_json_schema_tool_parameters(tool: &Tool, strict: Option<bool>) -> serde_json::Value {
    match strict {
        Some(true) => {
            make_strict_json_schema(&tool.parameters).unwrap_or_else(|_| tool.parameters.clone())
        }
        _ => tool.parameters.clone(),
    }
}

/// A resolved grammar-constrained tool definition.
#[derive(Clone, Debug)]
pub struct GrammarConstrainedSampling {
    pub format: GrammarFormat,
    pub definition: String,
    pub input_property: String,
}

/// Streaming buffer for a grammar tool's single string input property,
/// re-emitting JSON-encoded deltas.
#[derive(Default)]
pub struct GrammarToolInputJsonBuffer {
    input: String,
    started: bool,
    closed: bool,
}

pub fn get_grammar_tool_input(
    tool_name: &str,
    arguments: &serde_json::Map<String, serde_json::Value>,
    input_property: &str,
) -> Result<String, String> {
    match arguments.get(input_property) {
        Some(serde_json::Value::String(input)) => Ok(input.clone()),
        _ => Err(format!(
            "Grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string."
        )),
    }
}

impl GrammarToolInputJsonBuffer {
    pub fn append_json_delta(
        &mut self,
        input_property: &str,
        next_input: &str,
        close: bool,
    ) -> Result<Option<String>, String> {
        if self.closed {
            if close && next_input == self.input {
                return Ok(None);
            }
            return Err(format!(
                "grammar tool input for property \"{input_property}\" changed after it was closed"
            ));
        }
        if !next_input.starts_with(&self.input) {
            return Err(format!(
                "grammar tool input for property \"{input_property}\" changed non-monotonically"
            ));
        }

        let input_delta = &next_input[self.input.len()..];
        if !close && input_delta.is_empty() {
            return Ok(None);
        }

        let mut delta = String::new();
        if !self.started {
            // TS: `{${JSON.stringify(inputProperty)}:"` — the property keeps its quotes.
            delta += &format!(
                "{{{}:\"",
                serde_json::to_string(input_property).expect("string serializes")
            );
            self.started = true;
        }
        delta.push_str(&escape_json_string(input_delta));
        self.input = next_input.to_owned();

        if close {
            delta += "\"}";
            self.closed = true;
        }
        Ok(Some(delta))
    }
}

fn escape_json_string(text: &str) -> String {
    let encoded = serde_json::to_string(text).expect("strings always serialize");
    encoded[1..encoded.len() - 1].to_owned()
}

fn infer_grammar_input_property(tool: &Tool) -> Result<String, String> {
    let schema = &tool.parameters;
    if schema.get("type") != Some(&serde_json::Value::String("object".to_owned())) {
        return Err("grammar constrained sampling requires an object parameter schema".to_owned());
    }
    let required = schema
        .get("required")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            "grammar constrained sampling requires exactly one required string property".to_owned()
        })?;
    if required.len() != 1 || !required[0].is_string() {
        return Err(
            "grammar constrained sampling requires exactly one required string property".to_owned(),
        );
    }

    let input_property = required[0].as_str().expect("checked above");
    let property = schema
        .get("properties")
        .and_then(|value| value.get(input_property))
        .ok_or_else(|| {
            format!("grammar constrained sampling requires a properties entry for {input_property}")
        })?;
    if property.get("type") != Some(&serde_json::Value::String("string".to_owned())) {
        return Err(format!(
            "grammar constrained sampling property {input_property} must have type string"
        ));
    }
    Ok(input_property.to_owned())
}

/// Resolve whether a tool's json_schema constrained sampling should run in
/// strict mode: `Some(true)` when strict, `None` when unsupported/unused.
/// `strict: require` errors when strict mode is unavailable or the schema
/// cannot be made strict.
pub fn resolve_json_schema_strict_sampling(
    tool: &Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>, String> {
    let Some(crate::types::ConstrainedSamplingConfig::JsonSchema { strict }) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };

    if supports_strict_mode {
        return match make_strict_json_schema(&tool.parameters) {
            Ok(_) => Ok(Some(true)),
            Err(error) => {
                if *strict != StrictMode::Require {
                    return Ok(None);
                }
                Err(format!(
                    "Tool \"{}\" requires JSON-schema constrained sampling, but {}.",
                    tool.name, error.0
                ))
            }
        };
    }
    if *strict == StrictMode::Require {
        return Err(format!(
            "Tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported.",
            tool.name
        ));
    }
    Ok(None)
}

/// Resolve grammar-constrained sampling for a tool when the provider supports
/// OpenAI grammar tools.
pub fn resolve_grammar_constrained_sampling(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>, String> {
    let Some(crate::types::ConstrainedSamplingConfig::Grammar { variants }) =
        &tool.constrained_sampling
    else {
        return Ok(None);
    };

    if !supports_openai_grammar_tools {
        return Ok(None);
    }

    let lark_definition = variants.get(&GrammarFormat::OpenaiLark);
    let regex_definition = variants.get(&GrammarFormat::OpenaiRegex);
    let has_lark = lark_definition.is_some_and(|definition| !definition.trim().is_empty());
    let has_regex = regex_definition.is_some_and(|definition| !definition.trim().is_empty());
    if !has_lark && !has_regex {
        return Err(format!(
            "Tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided.",
            tool.name
        ));
    }

    let input_property = infer_grammar_input_property(tool).map_err(|error| {
        format!(
            "Tool \"{}\" cannot use grammar constrained sampling: {error}.",
            tool.name
        )
    })?;
    Ok(Some(GrammarConstrainedSampling {
        format: if has_lark {
            GrammarFormat::OpenaiLark
        } else {
            GrammarFormat::OpenaiRegex
        },
        definition: if has_lark {
            lark_definition.expect("checked above").clone()
        } else {
            regex_definition.expect("checked above").clone()
        },
        input_property,
    }))
}

/// Grammar input property per tool name, for message conversion.
pub fn create_grammar_tool_input_properties(
    tools: Option<&[Tool]>,
    supports_openai_grammar_tools: bool,
) -> Result<HashMap<String, String>, String> {
    let mut properties = HashMap::new();
    for tool in tools.unwrap_or_default() {
        if let Some(grammar) =
            resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)?
        {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConstrainedSamplingConfig;

    fn tool(parameters: serde_json::Value, config: Option<ConstrainedSamplingConfig>) -> Tool {
        Tool {
            name: "t".to_owned(),
            description: "d".to_owned(),
            parameters,
            constrained_sampling: config,
        }
    }

    #[test]
    fn strict_conversion_makes_optionals_nullable_and_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "integer" }
            },
            "required": ["a"]
        });
        let strict = make_strict_json_schema(&schema).unwrap();
        assert_eq!(strict["required"], serde_json::json!(["a", "b"]));
        assert_eq!(strict["additionalProperties"], serde_json::json!(false));
        assert_eq!(strict["properties"]["a"]["type"], "string");
        assert_eq!(
            strict["properties"]["b"]["anyOf"][1],
            serde_json::json!({ "type": "null" })
        );
    }

    #[test]
    fn strict_conversion_rejects_refs() {
        let schema = serde_json::json!({"type": "object", "$ref": "#/x"});
        assert!(make_strict_json_schema(&schema).is_err());
    }

    #[test]
    fn strict_require_errors_when_unsupported_schema() {
        let t = tool(
            serde_json::json!({"type": "object", "allOf": []}),
            Some(ConstrainedSamplingConfig::JsonSchema {
                strict: StrictMode::Require,
            }),
        );
        assert!(resolve_json_schema_strict_sampling(&t, true).is_err());
        assert!(resolve_json_schema_strict_sampling(&t, false).is_err());
    }

    #[test]
    fn strict_prefer_falls_back_when_unsupported_schema() {
        let t = tool(
            serde_json::json!({"type": "object", "allOf": []}),
            Some(ConstrainedSamplingConfig::JsonSchema {
                strict: StrictMode::Prefer,
            }),
        );
        assert_eq!(resolve_json_schema_strict_sampling(&t, true).unwrap(), None);
    }

    #[test]
    fn grammar_resolution_prefers_lark() {
        let mut variants = HashMap::new();
        variants.insert(GrammarFormat::OpenaiLark, "start: WORD".to_owned());
        variants.insert(GrammarFormat::OpenaiRegex, ".*".to_owned());
        let t = tool(
            serde_json::json!({
                "type": "object",
                "properties": { "input": { "type": "string" } },
                "required": ["input"]
            }),
            Some(ConstrainedSamplingConfig::Grammar { variants }),
        );
        let grammar = resolve_grammar_constrained_sampling(&t, true)
            .unwrap()
            .unwrap();
        assert_eq!(grammar.format, GrammarFormat::OpenaiLark);
        assert_eq!(grammar.input_property, "input");
    }

    #[test]
    fn grammar_buffer_emits_json_deltas() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let first = buffer
            .append_json_delta("input", "he", false)
            .unwrap()
            .unwrap();
        assert_eq!(first, "{\"input\":\"he");
        let second = buffer
            .append_json_delta("input", "hello", true)
            .unwrap()
            .unwrap();
        assert_eq!(second, "llo\"}");
        assert_eq!(
            buffer.append_json_delta("input", "hello", true).unwrap(),
            None
        );
        assert!(buffer.append_json_delta("input", "other", true).is_err());
    }

    #[test]
    fn grammar_buffer_escapes_json() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let delta = buffer
            .append_json_delta("in\"put", "a\"b", true)
            .unwrap()
            .unwrap();
        assert_eq!(delta, "{\"in\\\"put\":\"a\\\"b\"}");
    }

    #[test]
    fn grammar_buffer_integer_input() {
        let mut buffer = GrammarToolInputJsonBuffer::default();
        let delta = buffer
            .append_json_delta("code", "42", true)
            .unwrap()
            .unwrap();
        assert_eq!(delta, "{\"code\":\"42\"}");
    }
}

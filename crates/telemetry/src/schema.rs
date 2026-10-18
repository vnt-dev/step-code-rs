//! Serializable documentation for telemetry vocabularies.
//!
//! Attribute definitions are tagged enums: value sets and examples have the
//! same Rust type as the declared attribute, including empty arrays. Integer
//! and float schema types are explicit; legacy ambiguous "number" is unsupported.
//! These definitions describe data; they do not validate recording calls.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryAttributeCardinality {
    Low,
    High,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryAttributeMetadata {
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensitive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cardinality: Option<TelemetryAttributeCardinality>,
}

macro_rules! attribute_definitions {
    ($($variant:ident, $name:literal, $ty:ty, $allowed:ident, $examples:ty);* $(;)?) => {
        /// Each variant binds its value sets and examples to its attribute type.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        #[serde(tag = "type", rename_all_fields = "camelCase")]
        pub enum TelemetryAttributeDefinition {
            $(
                #[serde(rename = $name)]
                $variant {
                    #[serde(flatten)]
                    metadata: TelemetryAttributeMetadata,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    $allowed: Option<Vec<$ty>>,
                    #[serde(default, skip_serializing_if = "Option::is_none")]
                    examples: Option<Vec<$examples>>,
                },
            )*
        }
    };
}

attribute_definitions! {
    String, "string", String, values, String;
    Integer, "integer", i64, values, i64;
    UnsignedInteger, "unsigned_integer", u64, values, u64;
    Float, "float", f64, values, f64;
    Boolean, "boolean", bool, values, bool;
    StringArray, "string[]", String, element_values, Vec<String>;
    IntegerArray, "integer[]", i64, element_values, Vec<i64>;
    UnsignedIntegerArray, "unsigned_integer[]", u64, element_values, Vec<u64>;
    FloatArray, "float[]", f64, element_values, Vec<f64>;
    BooleanArray, "boolean[]", bool, element_values, Vec<bool>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryStartAttributeDefinition {
    #[serde(flatten)]
    pub definition: TelemetryAttributeDefinition,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEventAttributeDefinition {
    #[serde(flatten)]
    pub definition: TelemetryAttributeDefinition,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryEventDefinition {
    pub description: String,
    pub attributes: BTreeMap<String, TelemetryEventAttributeDefinition>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TelemetryParentDefinition {
    #[default]
    Any,
    RootOrExternal,
    Spans {
        spans: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetrySpanStatusDefault {
    #[default]
    Ok,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetrySpanStatusDefinition {
    pub default: TelemetrySpanStatusDefault,
    pub error_when: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetrySpanDefinition {
    pub description: String,
    pub parents: TelemetryParentDefinition,
    pub start_attributes: BTreeMap<String, TelemetryStartAttributeDefinition>,
    pub end_attributes: BTreeMap<String, TelemetryAttributeDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<BTreeMap<String, TelemetryEventDefinition>>,
    pub status: TelemetrySpanStatusDefinition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySchemaDefinition {
    pub version: u64,
    pub spans: BTreeMap<String, TelemetrySpanDefinition>,
}

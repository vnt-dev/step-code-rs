use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Primitive values and homogeneous arrays. Integers never pass through a
/// floating-point conversion. Tagged serialization preserves empty array types.
/// Float values are intended for actual floating-point measurements; use finite
/// values when serializing through formats such as JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum AttributeValue {
    String(String),
    Integer(i64),
    UnsignedInteger(u64),
    Float(f64),
    Boolean(bool),
    Strings(Vec<String>),
    Integers(Vec<i64>),
    UnsignedIntegers(Vec<u64>),
    Floats(Vec<f64>),
    Booleans(Vec<bool>),
}

pub type SpanAttributes = BTreeMap<String, AttributeValue>;

/// An exact empty attribute set for typed spans with no start/end attributes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoAttributes;

impl From<NoAttributes> for SpanAttributes {
    fn from(_: NoAttributes) -> Self {
        Self::new()
    }
}

impl From<f32> for AttributeValue {
    fn from(value: f32) -> Self {
        Self::Float(f64::from(value))
    }
}

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

macro_rules! value_from {
    ($($ty:ty => $variant:ident),* $(,)?) => {$(
        impl From<$ty> for AttributeValue {
            fn from(value: $ty) -> Self { Self::$variant(value) }
        }
    )*};
}

value_from! {
    String => String, i64 => Integer, u64 => UnsignedInteger, f64 => Float,
    bool => Boolean, Vec<String> => Strings, Vec<i64> => Integers,
    Vec<u64> => UnsignedIntegers, Vec<f64> => Floats, Vec<bool> => Booleans,
}

macro_rules! integer_from {
    ($($ty:ty => $wide:ty, $variant:ident),* $(,)?) => {$(
        impl From<$ty> for AttributeValue {
            fn from(value: $ty) -> Self { Self::$variant(<$wide>::from(value)) }
        }
    )*};
}

integer_from! {
    i8 => i64, Integer, i16 => i64, Integer, i32 => i64, Integer,
    u8 => u64, UnsignedInteger, u16 => u64, UnsignedInteger, u32 => u64, UnsignedInteger,
}

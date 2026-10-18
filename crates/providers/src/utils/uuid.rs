//! Standards-compliant, time-ordered UUIDv7 identifiers.

/// Generate an identifier ordered by creation within this process.
pub fn uuidv7() -> String {
    uuid::Uuid::now_v7().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encodes_current_timestamp_version_and_variant() {
        let before = crate::utils::now_ms();
        let id = uuid::Uuid::parse_str(&uuidv7()).unwrap();
        let after = crate::utils::now_ms();
        assert_eq!(id.get_version_num(), 7);
        assert_eq!(id.get_variant(), uuid::Variant::RFC4122);
        let (seconds, nanos) = id.get_timestamp().unwrap().to_unix();
        let millis = seconds * 1000 + u64::from(nanos) / 1_000_000;
        assert!((before..=after).contains(&millis));
    }
    #[test]
    fn identifiers_are_ordered() {
        let ids: Vec<_> = (0..1000).map(|_| uuidv7()).collect();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

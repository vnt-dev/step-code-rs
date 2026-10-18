//! Shared, typed YAML frontmatter parsing for resource loaders.

use serde::de::DeserializeOwned;

pub(super) fn parse<T: DeserializeOwned + Default>(
    content: &str,
    strip_bom: bool,
) -> Result<(T, String), String> {
    let content = if strip_bom {
        content.strip_prefix('\u{feff}').unwrap_or(content)
    } else {
        content
    };
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let Some(rest) = normalized.strip_prefix("---\n") else {
        return Ok((T::default(), normalized));
    };
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches('\n').trim_end() == "---" {
            let yaml = &rest[..offset];
            let metadata = if yaml.trim().is_empty() {
                T::default()
            } else {
                serde_saphyr::from_str(yaml).map_err(|error| error.to_string())?
            };
            return Ok((metadata, rest[offset + line.len()..].trim().to_owned()));
        }
        offset += line.len();
    }
    Err("Unterminated YAML frontmatter".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default, serde::Deserialize)]
    struct Metadata {
        description: Option<String>,
    }

    #[test]
    fn parses_block_scalars_and_escaped_strings() {
        let (metadata, body) =
            parse::<Metadata>("---\ndescription: |\n  first\n  second\n---\nbody", true).unwrap();
        assert_eq!(metadata.description.as_deref(), Some("first\nsecond\n"));
        assert_eq!(body, "body");
        let (metadata, _) =
            parse::<Metadata>("---\ndescription: \"a\\nb\" # comment\n---\n", true).unwrap();
        assert_eq!(metadata.description.as_deref(), Some("a\nb"));
    }

    #[test]
    fn rejects_malformed_yaml_and_requires_complete_fences() {
        assert!(parse::<Metadata>("---\ndescription: [\n---\n", true).is_err());
        assert!(parse::<Metadata>("---\ndescription: x\n---suffix", true).is_err());
        assert_eq!(
            parse::<Metadata>("---prefix\nbody", true).unwrap().1,
            "---prefix\nbody"
        );
    }
}

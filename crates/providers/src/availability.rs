//! Provider availability tri-state probe: "why can't I select this model?".

use crate::provider::ProviderProfile;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProviderAvailability {
    /// No credential resolved from the profile's auth ref.
    Unconfigured,
    /// Credential present, probe succeeded. `checked_at` is a Unix ms timestamp.
    Connected { checked_at: u64 },
    /// Credential present, probe failed; `reason` carries the redacted error.
    Error { reason: String, checked_at: u64 },
}

/// Resolves a profile's credential (from its `auth_ref`).
pub type CredentialResolver<'a> =
    Box<dyn Fn(&str) -> futures::future::BoxFuture<'a, Option<String>> + Send + 'a>;

/// Performs the connectivity probe for a configured profile; errors on failure.
pub type ProbeRequest<'a> = Box<
    dyn Fn(&ProviderProfile, &str) -> futures::future::BoxFuture<'a, Result<(), String>>
        + Send
        + 'a,
>;

/// Redact a resolved credential from an error reason so `reason` (and any
/// telemetry built from it) never carries the secret in plaintext.
fn redact_credential(reason: &str, credential: &str) -> String {
    if credential.is_empty() {
        return reason.to_owned();
    }
    reason.replace(credential, "[redacted]")
}

/// Tri-state availability probe:
/// 1. auth ref resolves nothing → `Unconfigured`;
/// 2. credential present and the probe succeeds → `Connected`;
/// 3. credential present and the probe fails → `Error` with the credential
///    redacted from the reason.
pub async fn probe_provider_availability<'a>(
    profile: &ProviderProfile,
    resolve_credential: CredentialResolver<'a>,
    send_probe: ProbeRequest<'a>,
    now: impl FnOnce() -> u64,
) -> ProviderAvailability {
    let Some(credential) = resolve_credential(&profile.auth_ref).await else {
        return ProviderAvailability::Unconfigured;
    };
    match send_probe(profile, &credential).await {
        Ok(()) => ProviderAvailability::Connected { checked_at: now() },
        Err(reason) => ProviderAvailability::Error {
            reason: redact_credential(&reason, &credential),
            checked_at: now(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderId;

    fn profile() -> ProviderProfile {
        ProviderProfile {
            id: ProviderId::new("test"),
            label: "Test".to_owned(),
            base_url: "https://api.test".to_owned(),
            auth_ref: "TEST_API_KEY".to_owned(),
            catalog: Vec::new(),
            default_api: None,
        }
    }

    #[tokio::test]
    async fn unconfigured_when_no_credential() {
        let availability = probe_provider_availability(
            &profile(),
            Box::new(|_| Box::pin(async { None })),
            Box::new(|_, _| Box::pin(async { Ok(()) })),
            || 42,
        )
        .await;
        assert_eq!(availability, ProviderAvailability::Unconfigured);
    }

    #[tokio::test]
    async fn connected_when_probe_succeeds() {
        let availability = probe_provider_availability(
            &profile(),
            Box::new(|_| Box::pin(async { Some("secret".to_owned()) })),
            Box::new(|_, _| Box::pin(async { Ok(()) })),
            || 100,
        )
        .await;
        assert_eq!(
            availability,
            ProviderAvailability::Connected { checked_at: 100 }
        );
    }

    #[tokio::test]
    async fn error_redacts_credential_from_reason() {
        let availability = probe_provider_availability(
            &profile(),
            Box::new(|_| Box::pin(async { Some("s3cret".to_owned()) })),
            Box::new(|_, _| Box::pin(async { Err("auth failed for s3cret".to_owned()) })),
            || 7,
        )
        .await;
        match availability {
            ProviderAvailability::Error { reason, checked_at } => {
                assert_eq!(reason, "auth failed for [redacted]");
                assert_eq!(checked_at, 7);
            }
            other => panic!("expected error state, got {other:?}"),
        }
    }
}

use step_harness_telemetry::testing::{TelemetryAdapterFixture, check_adapter_conformance};
use step_harness_telemetry::{InMemoryTelemetryContext, RecordedTelemetrySpan};

struct Fixture(InMemoryTelemetryContext);

impl TelemetryAdapterFixture for Fixture {
    type Context = InMemoryTelemetryContext;
    fn context(&self) -> &Self::Context {
        &self.0
    }
    async fn get_spans(&self) -> Vec<RecordedTelemetrySpan> {
        self.0.get_spans()
    }
}

#[tokio::test]
async fn in_memory_adapter_conforms_to_rust_contract() {
    check_adapter_conformance(|| async { Fixture(InMemoryTelemetryContext::new()) })
        .await
        .expect("adapter conformance");
}

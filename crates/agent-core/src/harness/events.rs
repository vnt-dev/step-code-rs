//! Harness lifecycle event bus: run start/end notifications.
//!
//! Ported from `@step-harness/agent-core` `src\harness\events.ts`.
//!
//! Notable differences from the TypeScript source:
//! - The `HarnessEvent` union becomes a Rust enum; `HarnessEventOfType<T>` is
//!   expressed by pattern matching rather than conditional types.
//! - Listeners are synchronous [`EventHandler`]s. The TypeScript listener may
//!   return a `Promise`, which `emit` invokes and discards (`void
//!   listener(event)`) because `emit` is synchronous. Rust listeners therefore
//!   cannot be awaited: an async listener must spawn its own task or forward
//!   the event over a channel from inside the callback.
//! - Listener registration is id-based, so [`HarnessEventBus::on`] can return
//!   an [`UnsubscribeFn`] that removes exactly that registration even if other
//!   listeners were added or removed in between.
//! - `emit` snapshots the listener list before invoking it, so listeners may
//!   subscribe, unsubscribe, or emit reentrantly without deadlocking.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Discriminant of a [`HarnessEvent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    RunStart,
    RunEnd,
}

impl EventType {
    /// Wire name of this event type, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::RunStart => "run_start",
            EventType::RunEnd => "run_end",
        }
    }
}

/// Outcome of a finished run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcomeKind {
    Completed,
    Aborted,
    Failed,
}

impl RunOutcomeKind {
    /// Wire name of this outcome, matching the TypeScript string union.
    pub fn as_str(&self) -> &'static str {
        match self {
            RunOutcomeKind::Completed => "completed",
            RunOutcomeKind::Aborted => "aborted",
            RunOutcomeKind::Failed => "failed",
        }
    }
}

/// A harness lifecycle event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarnessEvent {
    RunStart {
        lane: String,
        run_id: String,
    },
    RunEnd {
        lane: String,
        run_id: String,
        outcome: RunOutcomeKind,
        leaf_id: String,
    },
}

impl HarnessEvent {
    /// Discriminant of this event.
    pub fn event_type(&self) -> EventType {
        match self {
            HarnessEvent::RunStart { .. } => EventType::RunStart,
            HarnessEvent::RunEnd { .. } => EventType::RunEnd,
        }
    }

    /// Lane this event belongs to.
    pub fn lane(&self) -> &str {
        match self {
            HarnessEvent::RunStart { lane, .. } | HarnessEvent::RunEnd { lane, .. } => lane,
        }
    }

    /// Run id this event belongs to.
    pub fn run_id(&self) -> &str {
        match self {
            HarnessEvent::RunStart { run_id, .. } | HarnessEvent::RunEnd { run_id, .. } => run_id,
        }
    }
}

/// A synchronous event listener.
///
/// The TypeScript listener type is `(event) => void | Promise<void>`; because
/// `emit` is synchronous the returned promise is discarded. Rust listeners are
/// therefore plain synchronous callbacks. To run async work, spawn a task or
/// forward the event over a channel from inside the listener.
pub type EventHandler = Arc<dyn Fn(&HarnessEvent) + Send + Sync>;

/// Removes a previously registered listener when called.
pub type UnsubscribeFn = Box<dyn FnOnce() + Send + Sync>;

/// Listener registrations, keyed by an id assigned at registration time.
type ListenerMap = HashMap<usize, EventHandler>;

/// Shared listener registry behind the bus's mutex.
#[derive(Default)]
struct ListenerState {
    /// Direct listeners registered through [`HarnessEventBus::on`], by type.
    listeners: HashMap<EventType, ListenerMap>,
    /// Receivers registered through [`HarnessEventBus::watch`].
    watchers: ListenerMap,
    /// Source of registration ids.
    next_id: usize,
}

impl ListenerState {
    fn take_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Publishes [`HarnessEvent`]s to direct listeners and watchers.
///
/// The registry is held in an [`Arc`] so unsubscribe closures and watch handles
/// can keep it alive after the borrow taken by the registering call ends.
pub struct HarnessEventBus {
    shared: Arc<Mutex<ListenerState>>,
}

impl Default for HarnessEventBus {
    fn default() -> Self {
        HarnessEventBus {
            shared: Arc::new(Mutex::new(ListenerState::default())),
        }
    }
}

impl HarnessEventBus {
    pub fn new() -> Self {
        HarnessEventBus::default()
    }

    /// Register a listener for future events of one type and return its
    /// unsubscribe function.
    ///
    /// Earlier events are not replayed, and no snapshot or event buffer is
    /// provided; use [`HarnessEventBus::watch`] for both.
    pub fn on(&self, event_type: EventType, listener: EventHandler) -> UnsubscribeFn {
        let state = self.shared_state();
        let id = {
            let mut guard = state.lock().expect("event bus poisoned");
            let id = guard.take_id();
            guard
                .listeners
                .entry(event_type)
                .or_default()
                .insert(id, listener);
            id
        };
        Box::new(move || {
            let mut guard = state.lock().expect("event bus poisoned");
            if let Some(listeners) = guard.listeners.get_mut(&event_type) {
                listeners.remove(&id);
                if listeners.is_empty() {
                    guard.listeners.remove(&event_type);
                }
            }
        })
    }

    /// Publish an event to current event subscriptions and watch subscriptions.
    ///
    /// The listener lists are snapshotted before invocation, so a listener may
    /// subscribe, unsubscribe, or emit reentrantly without deadlocking. Async
    /// listeners cannot be awaited; see [`EventHandler`].
    pub fn emit(&self, event: &HarnessEvent) {
        let (typed, watchers) = {
            let guard = self.shared.lock().expect("event bus poisoned");
            let typed = guard
                .listeners
                .get(&event.event_type())
                .map(|listeners| listeners.values().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let watchers = guard.watchers.values().cloned().collect::<Vec<_>>();
            (typed, watchers)
        };

        // Deliver only to direct listeners registered for this event type.
        for listener in &typed {
            listener(event);
        }
        // Deliver every event to each watcher; `watch` handles buffering until
        // `start`.
        for listener in &watchers {
            listener(event);
        }
    }

    /// Register a watcher that buffers events until [`WatchHandle::start`].
    ///
    /// The snapshot is captured immediately, so it reflects state at
    /// registration time; events emitted afterwards are buffered and replayed
    /// in order once the listener starts.
    pub fn watch<TSnapshot>(
        &self,
        capture_snapshot: impl FnOnce() -> TSnapshot,
    ) -> WatchHandle<TSnapshot> {
        let phase: Arc<Mutex<WatchPhase>> =
            Arc::new(Mutex::new(WatchPhase::Buffering(VecDeque::new())));

        let phase_for_receive = Arc::clone(&phase);
        let receive: EventHandler = Arc::new(move |event: &HarnessEvent| {
            let mut phase = phase_for_receive.lock().expect("watch phase poisoned");
            match &mut *phase {
                // Unsubscribed watchers drop events.
                WatchPhase::Detached => {}
                WatchPhase::Buffering(buffer) => buffer.push_back(event.clone()),
                WatchPhase::Started(listener) => listener(event),
            }
        });

        let state = self.shared_state();
        let id = {
            let mut guard = state.lock().expect("event bus poisoned");
            let id = guard.take_id();
            guard.watchers.insert(id, receive);
            id
        };

        WatchHandle {
            snapshot: capture_snapshot(),
            phase,
            state,
            id,
        }
    }

    /// Shared handle to the listener registry, used by unsubscribe closures
    /// and watch handles so they can outlive the borrow taken by the
    /// registering call.
    fn shared_state(&self) -> Arc<Mutex<ListenerState>> {
        Arc::clone(&self.shared)
    }
}

/// Lifecycle phase of one watcher.
enum WatchPhase {
    /// Receiving and buffering events until `start` is called.
    Buffering(VecDeque<HarnessEvent>),
    /// Forwarding events to the listener passed to `start`.
    Started(EventHandler),
    /// Unsubscribed; events are dropped.
    Detached,
}

/// Handle returned by [`HarnessEventBus::watch`].
pub struct WatchHandle<TSnapshot> {
    /// Snapshot captured when the watch was registered.
    pub snapshot: TSnapshot,
    /// Buffered events, or the started listener.
    phase: Arc<Mutex<WatchPhase>>,
    /// Shared listener registry, used to detach this watcher.
    state: Arc<Mutex<ListenerState>>,
    /// Registration id of this watcher's receiver.
    id: usize,
}

impl<TSnapshot> WatchHandle<TSnapshot> {
    /// Start delivering events to `listener`, replaying buffered events first.
    ///
    /// The watcher stays in buffering mode while flushing, so reentrant
    /// emissions during the flush are buffered and replayed in order rather
    /// than delivered out of sequence.
    pub fn start(&self, listener: EventHandler) {
        loop {
            let pending: Vec<HarnessEvent> = {
                let mut phase = self.phase.lock().expect("watch phase poisoned");
                match &mut *phase {
                    WatchPhase::Detached => return,
                    WatchPhase::Buffering(buffer) => buffer.drain(..).collect(),
                    // Already started: replace the listener without replaying.
                    WatchPhase::Started(_) => {
                        *phase = WatchPhase::Started(listener);
                        return;
                    }
                }
            };
            if pending.is_empty() {
                break;
            }
            for event in &pending {
                listener(event);
            }
        }
        *self.phase.lock().expect("watch phase poisoned") = WatchPhase::Started(listener);
    }

    /// Stop receiving events and discard any buffered events.
    pub fn unsubscribe(self) {
        {
            let mut phase = self.phase.lock().expect("watch phase poisoned");
            *phase = WatchPhase::Detached;
        }
        let mut guard = self.state.lock().expect("event bus poisoned");
        guard.watchers.remove(&self.id);
    }

    /// Number of events currently buffered, for tests and diagnostics.
    pub fn buffered_len(&self) -> usize {
        match &*self.phase.lock().expect("watch phase poisoned") {
            WatchPhase::Buffering(buffer) => buffer.len(),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn run_start(lane: &str, run_id: &str) -> HarnessEvent {
        HarnessEvent::RunStart {
            lane: lane.to_owned(),
            run_id: run_id.to_owned(),
        }
    }

    fn run_end(lane: &str, run_id: &str, outcome: RunOutcomeKind) -> HarnessEvent {
        HarnessEvent::RunEnd {
            lane: lane.to_owned(),
            run_id: run_id.to_owned(),
            outcome,
            leaf_id: "leaf-1".to_owned(),
        }
    }

    #[test]
    fn delivers_events_to_typed_listeners() {
        let bus = HarnessEventBus::new();
        let starts = Arc::new(AtomicUsize::new(0));
        let ends = Arc::new(AtomicUsize::new(0));

        let starts_for_listener = Arc::clone(&starts);
        let _unsubscribe = bus.on(
            EventType::RunStart,
            Arc::new(move |_| {
                starts_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let ends_for_listener = Arc::clone(&ends);
        let _unsubscribe_ends = bus.on(
            EventType::RunEnd,
            Arc::new(move |_| {
                ends_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );

        bus.emit(&run_start("main", "r1"));
        bus.emit(&run_end("main", "r1", RunOutcomeKind::Completed));
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(ends.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unsubscribe_stops_delivery_and_removes_empty_type() {
        let bus = HarnessEventBus::new();
        let count = Arc::new(AtomicUsize::new(0));
        let count_for_listener = Arc::clone(&count);
        let unsubscribe = bus.on(
            EventType::RunStart,
            Arc::new(move |_| {
                count_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );

        bus.emit(&run_start("main", "r1"));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        unsubscribe();
        bus.emit(&run_start("main", "r2"));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(
            !bus.shared
                .lock()
                .unwrap()
                .listeners
                .contains_key(&EventType::RunStart)
        );
    }

    #[test]
    fn unsubscribing_one_listener_keeps_the_other() {
        let bus = HarnessEventBus::new();
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));

        let first_for_listener = Arc::clone(&first);
        let unsubscribe_first = bus.on(
            EventType::RunStart,
            Arc::new(move |_| {
                first_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let second_for_listener = Arc::clone(&second);
        let _keep = bus.on(
            EventType::RunStart,
            Arc::new(move |_| {
                second_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );

        unsubscribe_first();
        bus.emit(&run_start("main", "r1"));
        assert_eq!(first.load(Ordering::SeqCst), 0);
        assert_eq!(second.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn watch_buffers_until_start_and_replays_in_order() {
        let bus = HarnessEventBus::new();
        let handle = bus.watch(|| "snapshot");
        assert_eq!(handle.snapshot, "snapshot");
        assert_eq!(handle.buffered_len(), 0);

        bus.emit(&run_start("main", "r1"));
        bus.emit(&run_end("main", "r1", RunOutcomeKind::Completed));
        assert_eq!(handle.buffered_len(), 2);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_for_listener = Arc::clone(&seen);
        let seen_for_assertions = Arc::clone(&seen);
        handle.start(Arc::new(move |event| {
            seen_for_listener.lock().unwrap().push(event.clone());
        }));

        {
            let seen = seen_for_assertions.lock().unwrap();
            assert_eq!(seen.len(), 2);
            assert_eq!(seen[0].event_type(), EventType::RunStart);
            assert_eq!(seen[1].event_type(), EventType::RunEnd);
        }

        // Live events after start are delivered directly, not buffered.
        bus.emit(&run_start("main", "r2"));
        assert_eq!(seen_for_assertions.lock().unwrap().len(), 3);
        assert_eq!(handle.buffered_len(), 0);
    }

    #[test]
    fn watcher_sees_all_event_types() {
        let bus = HarnessEventBus::new();
        let handle = bus.watch(|| 0usize);
        bus.emit(&run_start("main", "r1"));
        bus.emit(&run_end("main", "r1", RunOutcomeKind::Aborted));
        assert_eq!(handle.buffered_len(), 2);
    }

    #[test]
    fn unsubscribe_discards_buffer_and_stops_delivery() {
        let bus = HarnessEventBus::new();
        let handle = bus.watch(|| ());
        bus.emit(&run_start("main", "r1"));
        assert_eq!(handle.buffered_len(), 1);

        handle.unsubscribe();
        // A second watcher confirms the bus still delivers to live watchers.
        let live = bus.watch(|| ());
        bus.emit(&run_start("main", "r2"));
        assert_eq!(live.buffered_len(), 1);
    }

    #[test]
    fn reentrant_emit_from_listener_does_not_deadlock() {
        let bus = Arc::new(HarnessEventBus::new());
        let count = Arc::new(AtomicUsize::new(0));

        let bus_for_listener = Arc::clone(&bus);
        let count_for_listener = Arc::clone(&count);
        let _unsubscribe = bus.on(
            EventType::RunStart,
            Arc::new(move |event| {
                count_for_listener.fetch_add(1, Ordering::SeqCst);
                // Reentrant emit of a different type from inside a listener.
                if event.run_id() == "r1" {
                    bus_for_listener.emit(&run_end("main", "r1", RunOutcomeKind::Completed));
                }
            }),
        );
        let ends = Arc::new(AtomicUsize::new(0));
        let ends_for_listener = Arc::clone(&ends);
        let _unsubscribe_ends = bus.on(
            EventType::RunEnd,
            Arc::new(move |_| {
                ends_for_listener.fetch_add(1, Ordering::SeqCst);
            }),
        );

        bus.emit(&run_start("main", "r1"));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(ends.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn event_accessors_expose_lane_and_run_id() {
        let start = run_start("alt", "r9");
        assert_eq!(start.lane(), "alt");
        assert_eq!(start.run_id(), "r9");
        let end = run_end("alt", "r9", RunOutcomeKind::Failed);
        assert_eq!(end.lane(), "alt");
        assert_eq!(end.run_id(), "r9");
        assert_eq!(end.event_type(), EventType::RunEnd);
        assert_eq!(RunOutcomeKind::Failed.as_str(), "failed");
        assert_eq!(EventType::RunEnd.as_str(), "run_end");
    }
}

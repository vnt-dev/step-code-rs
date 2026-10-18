//! App shell: wires the bridge commands and events to the UI.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  onApprovalRequest,
  onBridgeReady,
  onPlanReviewRequest,
  onSessionEvent,
  onRequestRecordUpdated,
} from "./api";
import MessageList from "./components/MessageList";
import Composer from "./components/Composer";
import Sidebar from "./components/Sidebar";
import TitleBar from "./components/TitleBar";
import TopBar from "./components/TopBar";
import ApprovalBar from "./components/ApprovalBar";
import PlanReviewBar from "./components/PlanReviewBar";
import SettingsPanel from "./components/SettingsPanel";
import { addApproval, addPlanReview, emptySessionView, hydrateSessionView, reduceSessionEvent, type SessionView } from "./lib/session-view";
import { loadImageDrafts, saveImageDrafts } from "./lib/draft-storage";
import { defaultDraft, draftKey, draftStorageKey, parseDrafts, supportedDraftThinkingLevel, type Draft } from "./lib/drafts";
import { groupProjects, projectKey } from "./lib/projects";
import type {
  Bootstrap,
  ImageBlock,
  SessionEvent,
  SessionSummary,
  ThinkingLevel,
} from "./types";

export default function App() {
  const [bootstrap, setBootstrap] = useState<Bootstrap | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const sessionViews = useRef(new Map<string, SessionView>());
  const [, rerender] = useState(0);
  const selectedSession = useRef<string | null>(null);
  const navigation = useRef(0);
  const refreshSequence = useRef(0);
  const snapshotsApplied = useRef(new Map<string, number>());
  const statsRequests = useRef(new Map<string, number>());
  const updateSession = useCallback((id: string, update: (current: SessionView) => SessionView) => {
    sessionViews.current.set(id, update(sessionViews.current.get(id) ?? emptySessionView()));
    rerender(value => value + 1);
  }, []);
  const [suggestion, setSuggestion] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draftPath, setDraftPath] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const recalling = useRef(new Set<string>());
  const [recallingSessions, setRecallingSessions] = useState(new Set<string>());
  /** Session being opened: the shell switched optimistically, transcript still in flight. */
  const [openingSession, setOpeningSession] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, Draft>>(() => {
    try { return parseDrafts(localStorage.getItem(draftStorageKey)); } catch { return {}; }
  });
  const draftSaveQueue = useRef(Promise.resolve());
  const [draftsLoaded, setDraftsLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void loadImageDrafts().then(saved => { if (!cancelled && saved) setDrafts(saved); })
      .catch(() => undefined).finally(() => { if (!cancelled) setDraftsLoaded(true); });
    return () => { cancelled = true; };
  }, []);
  const pendingSessions = useRef(new Map<string, string>());
  const isDraft = draftPath !== null || !bootstrap?.activeSession;
  const activeId = isDraft ? null : bootstrap?.activeSession?.id ?? null;
  const view = activeId ? sessionViews.current.get(activeId) ?? emptySessionView() : emptySessionView();
  const { messages, toolCalls, notices, stats, running, awaitingResponse } = view;
  const approval = view.approvals[0] ?? null;
  const planReview = view.planReviews[0] ?? null;
  const availableProjects = bootstrap ? groupProjects([...bootstrap.projects, bootstrap.cwd], sessions, "", bootstrap.removedProjects ?? []).map(group => group.path) : [];
  const selectedPath = draftPath ?? bootstrap?.activeSession?.cwd ?? availableProjects[0] ?? "";
  const composerKey = isDraft ? draftKey(selectedPath) : `session:${bootstrap?.activeSession?.id}`;
  const savedDraft = drafts[composerKey] ?? defaultDraft(bootstrap);
  const draftModel = bootstrap?.models.find(model => model.provider === savedDraft.provider && model.id === savedDraft.modelId);
  const draft = { ...savedDraft, thinkingLevel: supportedDraftThinkingLevel(draftModel, savedDraft.thinkingLevel) };
  const updateDraft = (patch: Partial<Draft>) => setDrafts(current => ({
    ...current, [composerKey]: { ...draft, ...current[composerKey], ...patch },
  }));
  useEffect(() => {
    if (!draftsLoaded) return;
    const projects = Object.fromEntries(Object.entries(drafts).filter(([key]) => key.startsWith("project:")));
    try { localStorage.setItem(draftStorageKey, JSON.stringify(Object.fromEntries(Object.entries(projects).map(([key, draft]) => [key, { ...draft, images: [] }])))); }
    catch { /* The IndexedDB copy can still preserve drafts. */ }
    draftSaveQueue.current = draftSaveQueue.current.then(() => saveImageDrafts(projects)).catch(() => {
      if (Object.values(projects).some(draft => draft.images?.length)) setError("图片草稿暂时无法保存到本地，关闭应用前请先发送。");
    });
  }, [drafts, draftsLoaded]);
  useEffect(() => {
    if (draftsLoaded && bootstrap && isDraft) setDrafts(current => {
      const saved = current[composerKey];
      if (saved?.provider || (saved && !bootstrap.models.some(model => model.configured))) return current;
      const defaults = defaultDraft(bootstrap);
      return { ...current, [composerKey]: saved ? { ...saved, provider: defaults.provider, modelId: defaults.modelId, thinkingLevel: defaults.thinkingLevel } : defaults };
    });
  }, [bootstrap, isDraft, composerKey, draftsLoaded]);
  const composerBootstrap = bootstrap && isDraft ? { ...bootstrap, cwd: selectedPath, activeSession: null,
    currentModel: bootstrap.models.find(model => model.provider === draft.provider && model.id === draft.modelId) ?? null,
    thinkingLevel: draft.thinkingLevel, approvalMode: draft.approvalMode, planActive: draft.planActive } : bootstrap;
  // Draft mode carries the plan switch locally; a session owns the real state.
  const planActive = Boolean(composerBootstrap?.planActive);

  const refreshBootstrap = useCallback(async (sessionId?: string) => {
    const epoch = navigation.current;
    const sequence = ++refreshSequence.current;
    const target = sessionId ?? selectedSession.current ?? undefined;
    const revision = target ? sessionViews.current.get(target)?.revision ?? 0 : 0;
    const next = await api.bootstrap(target);
    const id = next.activeSession?.id;
    if (id && sequence > (snapshotsApplied.current.get(id) ?? 0)) {
      snapshotsApplied.current.set(id, sequence);
      updateSession(id, current => hydrateSessionView(current, next, revision));
    }
    if (epoch !== navigation.current || sequence !== refreshSequence.current) return next;
    selectedSession.current = id ?? null;
    const live = id ? sessionViews.current.get(id) : undefined;
    const configuration = live && live.configurationRevision > revision ? live.configuration : {};
    setBootstrap({ ...next,
      planActive: configuration.planActive ?? next.planActive,
      planFilePath: configuration.planFilePath === undefined ? next.planFilePath : configuration.planFilePath,
      thinkingLevel: configuration.thinkingLevel ?? next.thinkingLevel,
      currentModel: configuration.modelId ? next.models.find(model => model.provider === configuration.provider && model.id === configuration.modelId) ?? null : next.currentModel,
    });
    setSessions(next.sessions);
    return next;
  }, [updateSession]);

  const applyEvent = useCallback((event: SessionEvent) => {
    updateSession(event.sessionId, current => reduceSessionEvent(current, event));
    if (event.type === "settled" || event.type === "messagesRecalled" || event.type === "compactionEnd" ||
      (event.type === "agent" && (event.event.type === "messageEnd" || event.event.type === "toolExecutionEnd"))) {
      const sequence = (statsRequests.current.get(event.sessionId) ?? 0) + 1;
      statsRequests.current.set(event.sessionId, sequence);
      void api.stats(event.sessionId).then(stats => {
        if (statsRequests.current.get(event.sessionId) === sequence) updateSession(event.sessionId, current => ({ ...current, stats }));
      }).catch(() => undefined);
    }
    if (event.type === "settled" || event.type === "messagesRecalled") {
      void api.listSessions().then(setSessions).catch(() => undefined);
    }
    if (event.type === "sessionInfoChanged" || event.type === "projectChanged") {
      void api.listSessions().then(setSessions).catch(() => undefined);
      if (event.type === "sessionInfoChanged") {
        setBootstrap(current => current?.activeSession?.id === event.sessionId
          ? { ...current, activeSession: { ...current.activeSession, name: event.name } }
          : current);
      }
    }
    if (event.type === "modelChanged" || event.type === "planModeChanged") {
      setBootstrap(current => {
        if (current?.activeSession?.id !== event.sessionId) return current;
        if (event.type === "planModeChanged") return { ...current, planActive: event.active, planFilePath: event.planFilePath };
        return { ...current, thinkingLevel: event.thinkingLevel, currentModel: current.models.find(model => model.provider === event.provider && model.id === event.modelId) ?? null };
      });
    }
  }, [updateSession]);

  // Initial load plus event subscriptions.
  useEffect(() => {
    let disposeSession: (() => void) | undefined;
    let disposeApproval: (() => void) | undefined;
    let disposePlanReview: (() => void) | undefined;
    let disposeReady: (() => void) | undefined;
    let disposeRequests: (() => void) | undefined;
    const requestTimers = new Map<string, ReturnType<typeof setTimeout>>();

    const boot = async () => {
      try {
        await refreshBootstrap();
      } catch (cause) {
        setError(String(cause));
      }
    };
    void boot();

    let disposed = false;
    const register = (subscription: Promise<() => void>, assign: (dispose: () => void) => void) => {
      void subscription.then(dispose => { if (disposed) dispose(); else assign(dispose); }).catch(cause => setError(String(cause)));
    };
    register(onBridgeReady(() => void refreshBootstrap().catch(cause => setError(String(cause)))), dispose => { disposeReady = dispose; });
    register(onRequestRecordUpdated(({ sessionId }) => {
      if (!sessionId) return;
      if (requestTimers.has(sessionId)) return;
      requestTimers.set(sessionId, setTimeout(() => {
        requestTimers.delete(sessionId);
        const sequence = (statsRequests.current.get(sessionId) ?? 0) + 1;
        statsRequests.current.set(sessionId, sequence);
        void api.stats(sessionId).then(stats => { if (!disposed && statsRequests.current.get(sessionId) === sequence) updateSession(sessionId, current => ({ ...current, stats })); }).catch(() => undefined);
      }, 500));
    }), dispose => { disposeRequests = dispose; });
    register(onSessionEvent(event => applyEvent(event)), dispose => { disposeSession = dispose; });
    register(onApprovalRequest(request => updateSession(request.sessionId, current => addApproval(current, request))), dispose => { disposeApproval = dispose; });
    register(onPlanReviewRequest(request => updateSession(request.sessionId, current => addPlanReview(current, request))), dispose => { disposePlanReview = dispose; });

    return () => {
      disposed = true;
      disposeSession?.();
      disposeApproval?.();
      disposePlanReview?.();
      disposeReady?.();
      disposeRequests?.();
      requestTimers.forEach(clearTimeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Open the session behind a draft and apply its saved configuration. Both
  // the send path and `/plan <task>` need a real session before they act.
  const ensureSession = useCallback(async () => {
    if (!isDraft) return activeId!;
    const pendingId = pendingSessions.current.get(composerKey);
    const id = pendingId ?? (await api.newSession(selectedPath)).id;
    pendingSessions.current.set(composerKey, id);
    if (pendingId) await api.openSession(id);
    await api.setModel(draft.provider, draft.modelId, id);
    await api.setThinkingLevel(draft.thinkingLevel as ThinkingLevel, id);
    await api.setApprovalMode(draft.approvalMode, id);
    await refreshBootstrap(id);
    return id;
  }, [isDraft, activeId, composerKey, selectedPath, draft, refreshBootstrap]);

  const send = useCallback(async (text: string, images: ImageBlock[], skills: string[]) => {
    const epoch = navigation.current;
    let id = activeId;
    setError(null);
    setCreating(true);
    try {
      id = await ensureSession();
      if (isDraft && draft.planActive) await api.enterPlanMode(id);
      updateSession(id, current => ({ ...current, running: true, awaitingResponse: true, runStartedAt: current.running ? current.runStartedAt : Date.now(), notices: [], error: null, revision: current.revision + 1, lifecycleRevision: current.revision + 1 }));
      await api.prompt(text, images, id, skills);
      if (isDraft) {
        setDrafts(current => ({ ...current, [composerKey]: { ...draft, text: "", images: [], skills: [], planActive: false } }));
        pendingSessions.current.delete(composerKey);
        if (epoch === navigation.current) { setDraftPath(null); setSuggestion(""); }
        void api.listSessions().then(setSessions).catch(() => undefined);
      }
    } catch (cause) {
      if (id) updateSession(id, current => ({ ...current, error: String(cause), running: false, awaitingResponse: false }));
      if (epoch === navigation.current) setError(String(cause));
      throw cause;
    } finally { setCreating(false); }
  }, [activeId, isDraft, composerKey, draft, ensureSession, updateSession]);

  // The composer's plan switch: before a session exists it is a local draft
  // flag (only the toolbar chip reacts); afterwards it drives the session.
  const togglePlanMode = useCallback(async () => {
    setError(null);
    if (isDraft) {
      updateDraft({ planActive: !planActive });
      return;
    }
    setCreating(true);
    try {
      const active = await api.togglePlanMode(activeId ?? undefined);
      setBootstrap(current => (current?.activeSession?.id === activeId ? { ...current, planActive: active } : current));
    } catch (cause) {
      setError(String(cause));
      throw cause;
    } finally {
      setCreating(false);
    }
  }, [activeId, isDraft, planActive, updateDraft]);

  // The toolbar chip only renders while plan mode is on, so it always exits.
  const exitPlanMode = useCallback(async () => {
    setError(null);
    if (isDraft) {
      updateDraft({ planActive: false });
      return;
    }
    setCreating(true);
    try {
      await api.exitPlanMode(activeId ?? undefined);
      setBootstrap(current => (current?.activeSession?.id === activeId ? { ...current, planActive: false } : current));
    } catch (cause) {
      setError(String(cause));
      throw cause;
    } finally {
      setCreating(false);
    }
  }, [activeId, isDraft, updateDraft]);

  const steer = useCallback(async (text: string, images: ImageBlock[], skills: string[]) => {
    if (!activeId) return;
    updateSession(activeId, current => ({ ...current, awaitingResponse: true }));
    try { await api.steer(text, images, activeId, skills); }
    catch (cause) {
      updateSession(activeId, current => ({ ...current, awaitingResponse: false }));
      throw cause;
    }
  }, [activeId, updateSession]);

  const recallLastUserMessage = useCallback(async (timestamp: number) => {
    if (!activeId || recalling.current.has(activeId)) return;
    const id = activeId;
    recalling.current.add(id);
    setRecallingSessions(new Set(recalling.current));
    try {
      await api.recallLastUserMessage(id, timestamp);
    } catch (cause) {
      updateSession(id, current => ({ ...current, error: String(cause) }));
    } finally {
      recalling.current.delete(id);
      setRecallingSessions(new Set(recalling.current));
    }
  }, [activeId, updateSession]);

  const decideApproval = useCallback(async (requestId: string, approved: boolean) => {
    if (!activeId) return;
    try {
      await api.respondApproval(requestId, approved);
      updateSession(activeId, current => ({ ...current, revision: current.revision + 1, approvalsRevision: current.revision + 1, approvals: current.approvals.filter(request => request.requestId !== requestId) }));
    } catch (cause) { updateSession(activeId, current => ({ ...current, error: String(cause) })); }
  }, [activeId, updateSession]);

  const decidePlanReview = useCallback(async (requestId: string, outcome: "approved" | "feedback" | "dismissed", feedback?: string) => {
    if (!activeId) return;
    try {
      await api.respondPlanReview(requestId, outcome, feedback);
      updateSession(activeId, current => ({ ...current, revision: current.revision + 1, planReviewsRevision: current.revision + 1, planReviews: current.planReviews.filter(request => request.requestId !== requestId) }));
    } catch (cause) { updateSession(activeId, current => ({ ...current, error: String(cause) })); }
  }, [activeId, updateSession]);

  // Switch the shell to the target session at once and let its transcript
  // arrive afterwards, so the sidebar never blocks on a loading state.
  const openSession = useCallback(async (sessionId: string) => {
    const epoch = ++navigation.current;
    const restoreDraft = draftPath;
    selectedSession.current = sessionId;
    setError(null);
    setOpeningSession(sessionId);
    setDraftPath(null);
    setSuggestion("");
    setSettingsOpen(false);
    setBootstrap(current => current
      ? { ...current, activeSession: sessions.find(session => session.id === sessionId) ?? current.activeSession }
      : current);
    try {
      await api.openSession(sessionId);
      if (epoch !== navigation.current) return;
      await refreshBootstrap(sessionId);
    } catch (cause) {
      if (epoch === navigation.current) {
        setError(String(cause));
        setDraftPath(restoreDraft);
      }
    } finally {
      if (epoch === navigation.current) setOpeningSession(null);
    }
  }, [refreshBootstrap, draftPath, sessions]);

  const newSession = useCallback(async (cwd?: string) => {
    ++navigation.current;
    setCreating(false);
    setOpeningSession(null);
    setError(null);
    setDraftPath(cwd ?? selectedPath);
    setSuggestion("");
    setSettingsOpen(false);
  }, [selectedPath]);

  const addProject = useCallback(async () => {
    try {
      const path = await api.addProject();
      if (path) { ++navigation.current; setDraftPath(path); setSuggestion(""); setSettingsOpen(false); await refreshBootstrap(); }
      return path;
    } catch (cause) {
      setError(String(cause));
      return null;
    }
  }, [refreshBootstrap]);

  const removeProject = async (path: string) => {
    try {
      await api.removeProject(path);
      if (projectKey(selectedPath) === projectKey(path)) {
        setDraftPath(availableProjects.find(project => projectKey(project) !== projectKey(path)) ?? "");
        setSuggestion("");
      }
      await refreshBootstrap();
    } catch (cause) {
      setError(String(cause));
      throw cause;
    }
  };

  const deleteSession = useCallback(
    async (sessionId: string) => {
      setError(null);
      try {
        await api.deleteSession(sessionId);
        sessionViews.current.delete(sessionId);
        if (selectedSession.current === sessionId) {
          ++navigation.current;
          selectedSession.current = null;
        }
        await refreshBootstrap();
      } catch (cause) {
        setError(String(cause));
        throw cause;
      }
    },
    [refreshBootstrap],
  );

  const renameSession = useCallback(
    async (sessionId: string, name: string) => {
      try {
        await api.renameSession(sessionId, name);
        await refreshBootstrap();
      } catch (cause) {
        setError(String(cause));
      }
    },
    [refreshBootstrap],
  );

  const setModel = useCallback(async (provider: string, modelId: string) => {
    try {
      await api.setModel(provider, modelId, activeId ?? undefined);
      await refreshBootstrap();
    } catch (cause) {
      setError(String(cause));
      throw cause;
    }
  }, [activeId, refreshBootstrap]);

  const setThinkingLevel = useCallback(async (level: ThinkingLevel) => {
    try {
      await api.setThinkingLevel(level, activeId ?? undefined);
      setBootstrap((current) => (current?.activeSession?.id === activeId ? { ...current, thinkingLevel: level } : current));
    } catch (cause) {
      setError(String(cause));
      throw cause;
    }
  }, [activeId]);

  const setApprovalMode = useCallback(async (mode: string) => {
    try {
      await api.setApprovalMode(mode, activeId ?? undefined);
      setBootstrap((current) => (current?.activeSession?.id === activeId ? { ...current, approvalMode: mode } : current));
    } catch (cause) {
      setError(String(cause));
      throw cause;
    }
  }, [activeId]);

  const saveSettings = useCallback(
    async (patch: Record<string, unknown>) => {
      await api.updateSettings(patch, activeId ?? undefined);
      await refreshBootstrap();
    },
    [activeId, refreshBootstrap],
  );

  return (
    <div className={`app${settingsOpen ? " settings-active" : ""}`}>
      <TitleBar />
      <Sidebar
        disabled={!bootstrap}
        settingsOpen={settingsOpen}
        onSettings={() => setSettingsOpen(true)}
        sessions={sessions}
        sessionActivity={Object.fromEntries([...sessionViews.current].flatMap<[string, "approval" | "running"]>(([id, state]) => state.approvals.length || state.planReviews.length
          ? [[id, "approval" as const]] : state.running ? [[id, "running" as const]] : []))}
        projects={bootstrap?.projects ?? []}
        removedProjects={bootstrap?.removedProjects ?? []}
        onAddProject={addProject}
        onRemoveProject={removeProject}
        activeId={isDraft ? null : bootstrap?.activeSession?.id ?? null}
        onOpen={openSession}
        onNew={newSession}
        onDelete={deleteSession}
        onRename={renameSession}
      />
      <main className="main">
        <div className="workspace-view" hidden={settingsOpen}>
        {!isDraft && <TopBar
          key={activeId}
          bootstrap={bootstrap}
          running={running}
          approvalPending={Boolean(approval)}
          runStartedAt={view.runStartedAt}
        />}
        {error || view.error ? (
          <div className="banner banner-error" onClick={() => { setError(null); if (activeId) updateSession(activeId, current => ({ ...current, error: null })); }}>
            {error ?? view.error}
          </div>
        ) : null}
        {bootstrap && bootstrap.models.length === 0 ? (
          <div className="banner banner-warning">
            连接模型服务后即可开始工作。 <button onClick={() => setSettingsOpen(true)}>配置供应商或登录 Step →</button>
          </div>
        ) : null}
        <div className={isDraft ? "draft-workspace" : "conversation-workspace"}>
        {isDraft ? <div className="draft-heading"><span className="eyebrow">BUILD WITH STEP</span><h1>有什么想让我帮忙的吗？</h1><p>选择项目，描述你的想法，从这里开始。</p></div> : <MessageList
          key={activeId}
          loading={openingSession === activeId}
          onRecall={recallLastUserMessage}
          recallDisabled={creating || recallingSessions.has(activeId!) || awaitingResponse}
          recalling={recallingSessions.has(activeId!)}
          onSuggestion={setSuggestion}
          messages={messages}
          toolCalls={toolCalls}
          thinkingDurations={view.thinkingDurations}
          thinkingStart={view.thinkingStart}
          runStartedAt={view.runStartedAt}
          notices={notices}
          hideThinking={Boolean(bootstrap?.settings.hideThinkingBlock)}
          awaitingResponse={awaitingResponse}
          running={running}
          executionTimings={view.executionTimings}
        />}
        {approval && !creating ? <ApprovalBar key={approval.requestId} request={approval} onDecide={decideApproval} /> : null}
        {planReview && !creating ? <PlanReviewBar key={planReview.requestId} request={planReview} onDecide={decidePlanReview} /> : null}
        <Composer
          key={composerKey}
          stats={isDraft ? null : stats}
          messages={isDraft ? [] : messages}
          executionTimings={isDraft ? [] : view.executionTimings}
          bootstrap={draftsLoaded ? composerBootstrap : null}
          text={draft.text}
          images={draft.images ?? []}
          selectedSkills={draft.skills ?? []}
          onSkillsChange={skills => updateDraft({ skills })}
          onImagesChange={images => updateDraft({ images })}
          onTextChange={text => updateDraft({ text })}
          draftMode={isDraft}
          projectPath={selectedPath}
          projects={availableProjects}
          onProjectChange={newSession}
          onAddProject={addProject}
          hasMessages={!isDraft && messages.length > 0}
          onModelChange={isDraft ? async (provider, modelId) => { updateDraft({ provider, modelId }); } : setModel}
          onThinkingChange={isDraft ? async thinkingLevel => { updateDraft({ thinkingLevel }); } : setThinkingLevel}
          onApprovalChange={isDraft ? async approvalMode => { updateDraft({ approvalMode }); } : setApprovalMode}
          onPlanToggle={togglePlanMode}
          onPlanExit={exitPlanMode}
          planActive={planActive}
          onCompact={() => api.compact(activeId ?? undefined)}
          suggestion={suggestion}
          running={running}
          disabled={!draftsLoaded || creating || recallingSessions.has(activeId!) || !selectedPath || !composerBootstrap?.currentModel?.configured}
          onSend={send}
          onSteer={steer}
          onFollowUp={(text, images, skills) => api.followUp(text, images, activeId ?? undefined, skills)}
          onAbort={() => { if (activeId) { updateSession(activeId, current => ({ ...current, awaitingResponse: false })); void api.abort(activeId).catch(cause => updateSession(activeId, current => ({ ...current, error: String(cause) }))); } }}
        />
        </div>
        </div>
        {bootstrap && <SettingsPanel bootstrap={bootstrap} open={settingsOpen} onSave={saveSettings} onRefresh={async () => { await refreshBootstrap(); }} onClose={() => setSettingsOpen(false)} />}
      </main>

    </div>
  );
}

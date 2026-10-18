import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Bootstrap, SkillSummary, ThinkingLevel, ImageBlock, AgentMessage, ExecutionTiming, SessionStats } from "../types";
import { api } from "../api";
import ContextUsageIndicator from "./ContextUsageIndicator";
import RoundSpeed from "./RoundSpeed";
import Dropdown, { type DropdownOption } from "./Dropdown";
import Icon from "./Icon";
import { folderName, projectKey } from "../lib/projects";
import { imageSource, imageTypes, maxImages, readImage, type ImageAttachment } from "../lib/images";
interface Props {
  stats: SessionStats | null; messages: AgentMessage[]; executionTimings: ExecutionTiming[];
  images: ImageAttachment[]; onImagesChange: (images: ImageAttachment[]) => void;
  selectedSkills: SkillSummary[]; onSkillsChange: (skills: SkillSummary[]) => void;
  text: string; onTextChange: (text: string) => void;
  draftMode: boolean; projectPath: string; projects: string[];
  onProjectChange: (path: string) => Promise<void>; onAddProject: () => Promise<string | null>;
  bootstrap: Bootstrap | null; running: boolean; disabled: boolean; suggestion: string; hasMessages: boolean;
  onSend: (text: string, images: ImageBlock[], skills: string[]) => Promise<void>; onSteer: (text: string, images: ImageBlock[], skills: string[]) => Promise<void>; onFollowUp: (text: string, images: ImageBlock[], skills: string[]) => Promise<void>; onAbort: () => void;
  onModelChange: (provider: string, modelId: string) => Promise<void>;
  onThinkingChange: (level: ThinkingLevel) => Promise<void>; onApprovalChange: (mode: string) => Promise<void>;
  /** Toggle plan mode: a local draft flag before a session exists. */
  onPlanToggle: () => Promise<void>;
  /** Leave plan mode (the toolbar chip only shows while it is on). */
  onPlanExit: () => Promise<void>;
  planActive: boolean;
  onCompact: () => Promise<void>;
}
const permissions: DropdownOption[] = [
  { value: "ask", label: "变更前确认", description: "修改文件或执行命令前询问我" },
  { value: "bypass", label: "自主模式", description: "自动执行普通操作，在有风险时询问" },
  { value: "autopilot", label: "完全访问", description: "自动执行工具和命令，无需审批" },
];
const thinking: DropdownOption[] = [
  { value: "off", label: "关闭思考", description: "直接生成回答" },
  { value: "minimal", label: "最少", description: "以响应速度为优先" },
  { value: "low", label: "较低", description: "适合简单修改" },
  { value: "medium", label: "标准", description: "兼顾速度和推理深度" },
  { value: "high", label: "深入", description: "适合复杂分析和实现" },
  { value: "xhigh", label: "更深入", description: "投入更多推理时间" },
  { value: "max", label: "最高", description: "使用模型支持的最大推理力度" },
];
export default function Composer({ stats, messages, executionTimings, images, onImagesChange, selectedSkills, onSkillsChange, text, onTextChange: setText, draftMode, projectPath, projects, onProjectChange, onAddProject, bootstrap, running, disabled, suggestion, hasMessages, onSend, onSteer, onFollowUp, onAbort, onModelChange, onThinkingChange, onApprovalChange, onPlanToggle, onPlanExit, planActive, onCompact }: Props) {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [skillsError, setSkillsError] = useState("");
  const skillRequest = useRef(0);
  const refreshSkills = async () => {
    if (!projectPath || !bootstrap) return;
    const request = ++skillRequest.current;
    setSkillsLoading(true); setSkillsError("");
    try { const next = await api.skills(projectPath); if (request === skillRequest.current) setSkills(next); }
    catch (cause) { if (request === skillRequest.current) { setSkills([]); setSkillsError(String(cause)); } }
    finally { if (request === skillRequest.current) setSkillsLoading(false); }
  };
  useEffect(() => {
    setSkills([]);
    void refreshSkills();
    return () => { ++skillRequest.current; };
  }, [projectPath, Boolean(bootstrap)]);
  const [submitting, setSubmitting] = useState(false);
  const [configuring, setConfiguring] = useState(false);
  const [error, setError] = useState("");
  const [loadingImages, setLoadingImages] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (draftMode && bootstrap) input.current?.focus(); }, [draftMode, Boolean(bootstrap)]);
  useEffect(() => { if (suggestion) { setText(suggestion); input.current?.focus(); } }, [suggestion]);
  const submit = async (followUp = false) => {
    const trimmed = text.trim();
    if ((!trimmed && !images.length && !selectedSkills.length) || disabled || submitting || configuring || loadingImages) return;
    if (images.length && !bootstrap?.currentModel?.supportsImages) { setError("当前模型不支持图片，请选择支持图片的模型。"); return; }
    setSubmitting(true); setError("");
    try { await (followUp ? onFollowUp(trimmed, images, selectedSkills.map(skill => skill.filePath)) : running ? onSteer(trimmed, images, selectedSkills.map(skill => skill.filePath)) : onSend(trimmed, images, selectedSkills.map(skill => skill.filePath))); setText(""); onImagesChange([]); onSkillsChange([]); }
    catch (e) { setError(String(e)); }
    finally { setSubmitting(false); }
  };
  const addImages = async (files: File[], pasted = false) => {
    if (!files.length || submitting || loadingImages || !bootstrap) return;
    if (images.length + files.length > maxImages) { setError(`最多可添加 ${maxImages} 张图片。`); return; }
    setLoadingImages(true); setError("");
    try { onImagesChange([...images, ...await Promise.all(files.map(file => readImage(file, pasted)))]); }
    catch (cause) { setError(String(cause)); }
    finally { setLoadingImages(false); }
  };
  const configure = async (action: () => Promise<void>) => {
    setConfiguring(true);
    try { await action(); }
    catch (cause) { setError(String(cause)); throw cause; }
    finally { setConfiguring(false); }
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit(); }
  };
  const current = bootstrap?.currentModel;
  const thinkingOptions = thinking.filter(level => current?.supportedThinkingLevels.includes(level.value as ThinkingLevel));
  const locked = running || submitting || configuring || loadingImages || !bootstrap;
  const projectOptions: DropdownOption[] = [...new Map(projects.filter(Boolean).map(path => [projectKey(path), path])).values()].map(path => ({
    value: projectKey(path), label: folderName(path), description: path, group: "项目", icon: <Icon name="folder" />,
  }));
  projectOptions.push({ value: "add-project", label: "打开文件夹…", description: "添加项目并开始编辑草稿", group: "添加项目", icon: <Icon name="plus" /> });
  const models: DropdownOption[] = [...(bootstrap?.models ?? [])].sort((a, b) => a.provider.localeCompare(b.provider) || a.name.localeCompare(b.name)).map(model => ({
    value: JSON.stringify([model.provider, model.id]), label: model.name || model.id, group: model.provider,
    description: `${model.id} · ${Math.round(model.contextWindow / 1000)}k 上下文`, disabled: !model.configured,
  }));
  const actions: DropdownOption[] = [
    { group: "添加", value: "image", label: "添加图片…", description: "选择图片，也可粘贴或拖入", icon: <Icon name="image" />, disabled: submitting || loadingImages || !bootstrap },
    ...(running ? [{ value: "queue", label: "排队发送当前输入", description: "当前任务完成后继续执行", icon: <Icon name="queue" />, disabled: disabled || (!text.trim() && !images.length && !selectedSkills.length) || submitting || loadingImages }] : []),
    { value: "plan", label: planActive ? "退出计划模式" : "进入计划模式", description: planActive ? "恢复文件修改，按已批准的计划执行" : "先调研并产出方案，经你确认后再执行", icon: <Icon name="plan" />, disabled: locked },
    { value: "compact", label: "压缩对话上下文", description: "保留关键信息，腾出上下文空间", icon: <Icon name="compact" />, disabled: locked || !hasMessages },
    ...skills.map(skill => ({ value: `skill:${skill.filePath}`, label: skill.name, description: skill.description,
      group: "技能", icon: <Icon name="skill" />, disabled: submitting || !bootstrap || selectedSkills.some(selected => selected.filePath === skill.filePath) })),
    ...(!skills.length ? [{ value: "skills-empty", label: skillsLoading ? "正在加载技能…" : skillsError ? "技能加载失败" : "暂无可用技能",
      description: skillsError || "将 SKILL.md 放入项目 .step-rs/skills 或用户技能目录", group: "技能", disabled: true, icon: <Icon name="skill" /> }] : []),
  ];
  return <div className="composer">
    {error && <p className="error-text" role="alert">{error}</p>}
    {draftMode && <div className="composer-project-row"><Dropdown label="选择项目" trigger={<><Icon name="folder" /><span>{folderName(projectPath) || "选择项目"}</span></>} className="project-control" value={projectKey(projectPath)} options={projectOptions} searchable placement="auto" searchPlaceholder="搜索项目名称或路径…" emptyText="没有匹配的项目" disabled={locked} onSelect={async value => {
      if (value === "add-project") { await onAddProject(); }
      else { const path = projects.find(path => projectKey(path) === value); if (path) await onProjectChange(path); }
    }} /></div>}
    <div className="composer-shell" onDragOver={event => { if (event.dataTransfer.types.includes("Files")) event.preventDefault(); }} onDrop={event => {
      if (event.dataTransfer.files.length) { event.preventDefault(); void addImages(Array.from(event.dataTransfer.files)); }
    }}>
      <input ref={fileInput} type="file" hidden accept={imageTypes.join(",")} multiple onChange={event => { void addImages(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
      {images.length > 0 && <div className="composer-images">{images.map(image => <div className="composer-image" key={image.id}>
        <img src={imageSource(image)} alt={image.pasted ? "附加图片" : image.name} />
        {!image.pasted && <span title={image.name}>{image.name}</span>}
        <button aria-label={image.pasted ? "移除图片" : `移除图片 ${image.name}`} disabled={submitting || loadingImages} onClick={() => onImagesChange(images.filter(item => item.id !== image.id))}>×</button>
      </div>)}</div>}
      {images.length > 0 && !current?.supportsImages && <p className="error-text" role="alert">当前模型不支持图片，请切换支持图片的模型。</p>}
      {selectedSkills.length > 0 && <div className="composer-skills" aria-label="已选择的技能">{selectedSkills.map(skill =>
        <span className="skill-chip" key={skill.filePath} title={`${skill.description}\n${skill.filePath}`}><Icon name="skill" /><span>{skill.name}</span>
          <button type="button" aria-label={`移除技能 ${skill.name}`} disabled={submitting} onClick={() => onSkillsChange(selectedSkills.filter(item => item.filePath !== skill.filePath))}>×</button>
        </span>)}</div>}
      <textarea ref={input} aria-label="任务描述" disabled={submitting || loadingImages || !bootstrap} value={text} onPaste={event => {
        const files = Array.from(event.clipboardData.items).filter(item => item.kind === "file" && item.type.startsWith("image/")).map(item => item.getAsFile()).filter((file): file is File => file !== null);
        if (files.length) { event.preventDefault(); void addImages(files, true); }
      }} onChange={e => setText(e.target.value)} onKeyDown={onKeyDown} placeholder={running ? "补充要求，调整当前任务的执行方向…" : "描述你想完成的任务…"} rows={3} />
      <div className="composer-toolbar">
        <div className="composer-tools-left">
          <Dropdown label="更多操作" trigger={<Icon name="plus" />} actions className="composer-more" options={actions} onOpen={() => void refreshSkills()} restoreFocusOnSelect={false} onSelect={async value => {
            if (value === "image") fileInput.current?.click();
            else if (value === "queue") await submit(true);
            else if (value === "plan") await configure(() => onPlanToggle());
            else if (value.startsWith("skill:")) {
              const skill = skills.find(item => item.filePath === value.slice(6));
              if (skill && !selectedSkills.some(item => item.filePath === skill.filePath)) onSkillsChange([...selectedSkills, skill]);
              requestAnimationFrame(() => input.current?.focus());
            }
            else if (value === "compact") await configure(() => onCompact());
          }} />
          <Dropdown label="权限模式" trigger={<><Icon name="shield" /><span>{permissions.find(p => p.value === bootstrap?.approvalMode)?.label || "变更前确认"}</span></>} className={`permission-control ${bootstrap?.approvalMode === "autopilot" ? "elevated" : ""}`} value={bootstrap?.approvalMode ?? "ask"} options={permissions} disabled={locked} onSelect={value => configure(() => onApprovalChange(value))} />
          {planActive && <button type="button" className="plan-chip" title="退出计划模式" disabled={locked} onClick={() => configure(() => onPlanExit())}><Icon name="bulb" className="plan-chip-idle" /><Icon name="closeCircle" className="plan-chip-exit" /><span>计划</span></button>}
        </div>
        <div className="composer-tools-right">
          {!draftMode && <RoundSpeed messages={messages} stats={stats} running={running} />}
          {!draftMode && <ContextUsageIndicator messages={messages} stats={stats} running={running} executionTimings={executionTimings} maxContext={current?.contextWindow ?? 0} />}
          <Dropdown label="选择模型" trigger={<span className="model-control-label">{current?.name || current?.id || "选择模型"}</span>} className="model-control" value={current ? JSON.stringify([current.provider, current.id]) : undefined} options={models} searchable align="end" disabled={locked || !models.length} onSelect={value => {
            const model = bootstrap?.models.find(m => JSON.stringify([m.provider, m.id]) === value);
            return model ? configure(() => onModelChange(model.provider, model.id)) : undefined;
          }} />
          <Dropdown label="思考级别" trigger={<><Icon name="brain" /><span>{current?.reasoning ? thinkingOptions.find(level => level.value === bootstrap?.thinkingLevel)?.label || "未选择" : "无思考"}</span></>} className="thinking-control" value={bootstrap?.thinkingLevel ?? "off"} options={thinkingOptions} align="end" disabled={locked || !current?.reasoning || !thinkingOptions.length} onSelect={value => configure(() => onThinkingChange(value as ThinkingLevel))} />
          <button className="primary composer-send" aria-label={running ? "发送补充要求" : "发送消息"} title={running ? "发送补充要求" : "发送消息 (Enter)"} disabled={disabled || submitting || configuring || loadingImages || (!text.trim() && !images.length && !selectedSkills.length) || (images.length > 0 && !current?.supportsImages)} onClick={() => void submit()}><Icon name="arrow" /></button>
          {running && <button className="composer-stop" aria-label="停止任务" title="停止任务" onClick={onAbort}><Icon name="stop" /></button>}
        </div>
      </div>
    </div>
    <div className="composer-footnote"><span>{running ? "Enter 补充要求 · 加号菜单可排队发送" : "Enter 发送 · Shift + Enter 换行 · 可粘贴或拖入图片"}</span><span>{loadingImages ? "正在读取图片…" : configuring ? "正在更新配置…" : running ? "任务执行中" : draftMode ? "" : "本地工作空间"}</span></div>
  </div>;
}

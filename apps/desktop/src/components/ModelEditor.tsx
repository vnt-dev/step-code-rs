import { useRef, useState } from "react";
import type { ModelDraft } from "../types";
import { modelThinkingLevels, parseThinkingLevels, parseThinkingMap, parseTokenCount } from "../lib/model-form";
import SettingsModal from "./SettingsModal";

interface Props { model: ModelDraft; editing: boolean; busy: boolean; error: string; onClose: () => void; onSave: (model: ModelDraft) => Promise<void>; }
export default function ModelEditor({ model, editing, busy, error, onClose, onSave }: Props) {
  const [id, setId] = useState(model.id);
  const [name, setName] = useState(model.name ?? "");
  const [context, setContext] = useState(String(model.contextWindow ?? 200000));
  const [max, setMax] = useState(String(model.maxTokens ?? 8192));
  const [image, setImage] = useState(model.input?.includes("image") ?? false);
  const [levels, setLevels] = useState<string[]>(modelThinkingLevels(model));
  const levelInputs = useRef<(HTMLInputElement | null)[]>([]);
  const [mapping, setMapping] = useState(model.thinkingLevelMap ? JSON.stringify(model.thinkingLevelMap, null, 2) : "");
  const [localError, setLocalError] = useState("");
  const reset = () => { setId(model.id); setName(model.name ?? ""); setContext(String(model.contextWindow ?? 200000)); setMax(String(model.maxTokens ?? 8192)); setImage(model.input?.includes("image") ?? false); setLevels(modelThinkingLevels(model)); setMapping(model.thinkingLevelMap ? JSON.stringify(model.thinkingLevelMap, null, 2) : ""); setLocalError(""); };
  const normalize = (value: string, set: (value: string) => void) => { try { set(String(parseTokenCount(value))); } catch { /* Keep incomplete input editable; submit reports validation. */ } };
  const preview = (value: string) => { try { return `${parseTokenCount(value).toLocaleString("en-US")} Token`; } catch { return "支持 k / m，1k = 1,000，1m = 1,000,000"; } };
  return <SettingsModal title={editing ? "编辑模型" : "添加模型"} busy={busy} onClose={onClose}>
    <form onSubmit={event => { event.preventDefault(); setLocalError(""); void (async () => {
      try {
        const contextWindow = parseTokenCount(context), maxTokens = parseTokenCount(max);
        if (maxTokens > contextWindow) throw new Error("最大输出 Token 不能超过上下文窗口。");
        const selectedLevels = parseThinkingLevels(levels);
        const next = { ...model, id: id.trim(), name: name.trim() || undefined, contextWindow, maxTokens,
          input: image ? ["text", "image"] as ("text" | "image")[] : ["text"] as ("text" | "image")[],
          reasoning: selectedLevels.length > 0, thinkingLevelMap: parseThinkingMap(mapping, selectedLevels) };
        if (!next.id) throw new Error("请输入模型 ID。");
        await onSave(next);
      } catch (cause) { setLocalError(cause instanceof Error ? cause.message : String(cause)); }
    })(); }}>
      <div className="settings-modal-body model-editor">
        {(localError || error) && <div className="banner banner-error" role="alert">{localError || error}</div>}
        <label>模型 ID<input autoFocus required disabled={busy} value={id} placeholder="模型 ID" onChange={event => setId(event.target.value)} /></label>
        <label>显示名称<input disabled={busy} value={name} placeholder="留空使用模型 ID" onChange={event => setName(event.target.value)} /></label>
        <label>上下文窗口<input required disabled={busy} value={context} placeholder="例如 128k 或 1m" onChange={event => setContext(event.target.value)} onBlur={() => normalize(context, setContext)} /><small>{preview(context)}</small></label>
        <label>最大输出 Token<input required disabled={busy} value={max} placeholder="例如 8k" onChange={event => setMax(event.target.value)} onBlur={() => normalize(max, setMax)} /><small>{preview(max)}</small></label>
        <details open className="model-advanced"><summary>高级配置</summary>
          <fieldset disabled={busy}><legend>输入类型</legend><div className="model-chips">
            <label className="model-chip"><input type="checkbox" checked disabled />文本</label>
            <label className="model-chip"><input type="checkbox" checked={image} onChange={event => setImage(event.target.checked)} />图片</label>
            <label className="model-chip unavailable" title="当前消息协议暂不支持视频输入"><input type="checkbox" disabled />视频</label>
            <label className="model-chip unavailable" title="当前消息协议暂不支持原生 PDF 输入"><input type="checkbox" disabled />PDF</label>
          </div></fieldset>
          <fieldset disabled={busy}><legend>推理等级（从低到高）</legend><div className="model-chips">
            {levels.map((level, index) => <div className="model-level-input" key={index}>
              <input ref={element => { levelInputs.current[index] = element; }} aria-label={`推理等级 ${index + 1}`} value={level} placeholder="如 low" onChange={event => setLevels(levels.map((item, i) => i === index ? event.target.value : item))} />
              <button type="button" aria-label={`移除推理等级 ${index + 1}`} onClick={() => setLevels(levels.filter((_, i) => i !== index))}>×</button>
            </div>)}
            <button type="button" className="model-chip" aria-label="添加推理等级" onClick={() => {
              const index = levels.length;
              setLevels([...levels, ""]);
              requestAnimationFrame(() => levelInputs.current[index]?.focus());
            }}>＋</button>
          </div><small>手动填写，如 low、medium、high；空白项不保存，不填写表示无思考。</small></fieldset>
          <label>推理参数映射<textarea rows={5} disabled={busy} value={mapping} placeholder={'{\n  "low": "low",\n  "medium": "medium",\n  "high": "high"\n}'} onChange={event => setMapping(event.target.value)} /><small>JSON 对象，将已选等级映射为服务端参数值；留空使用同名值。</small></label>
        </details>
      </div>
      <footer className="settings-modal-footer"><button type="button" className="reset-form" disabled={busy} onClick={reset}>重置表单</button><button type="button" disabled={busy} onClick={onClose}>取消</button><button className="primary" disabled={busy}>{busy ? "保存中…" : "保存"}</button></footer>
    </form>
  </SettingsModal>;
}

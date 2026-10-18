import { useState } from "react";
import type { ModelDraft, ProviderDraft } from "../types";
import Dropdown from "./Dropdown";
import ModelEditor from "./ModelEditor";
import SettingsModal from "./SettingsModal";

export default function ProviderEditor({ initial, editing, busy, error, onClose, onSave }: {
  initial: ProviderDraft; editing: boolean; busy: boolean; error: string;
  onClose: () => void; onSave: (draft: ProviderDraft) => Promise<void>;
}) {
  const [draft, setDraft] = useState(initial);
  const [modelEdit, setModelEdit] = useState<{ model: ModelDraft; index: number } | null>(null);
  const protocols = [
    { value: "openai-completions", label: "OpenAI Chat Completions" },
    { value: "openai-responses", label: "OpenAI Responses" },
    { value: "anthropic-messages", label: "Anthropic Messages" },
  ];
  return <>
    {!modelEdit && <SettingsModal title={editing ? "编辑供应商" : "添加供应商"} busy={busy} onClose={onClose}>
      <form onSubmit={event => { event.preventDefault(); void onSave({ ...draft, id: draft.id.trim(), name: draft.name.trim(), baseUrl: draft.baseUrl.trim() }); }}>
        <div className="settings-modal-body model-editor">
          {error && <div className="banner banner-error" role="alert">{error}</div>}
          <label>供应商 ID<input autoFocus required pattern="[A-Za-z0-9_-]+" disabled={editing || busy} value={draft.id} placeholder="例如 my-provider" onChange={event => setDraft({ ...draft, id: event.target.value })} /></label>
          <label>显示名称<input required disabled={busy} value={draft.name} placeholder="供应商名称" onChange={event => setDraft({ ...draft, name: event.target.value })} /></label>
          <label>API 地址<input required type="url" disabled={busy} value={draft.baseUrl} placeholder="https://api.example.com/v1" onChange={event => setDraft({ ...draft, baseUrl: event.target.value })} /></label>
          <label>API 协议<Dropdown label="API 协议" className="form-select" placement="auto" value={draft.api} disabled={busy} trigger={<span>{protocols.find(item => item.value === draft.api)?.label ?? draft.api}</span>} options={protocols} onSelect={api => setDraft({ ...draft, api })} /></label>
          <label>API 密钥<input type="password" autoComplete="new-password" disabled={busy} value={draft.apiKey ?? ""} placeholder={editing ? "留空保留已有密钥" : "输入密钥，或使用环境变量"} onChange={event => setDraft({ ...draft, apiKey: event.target.value || undefined })} /></label>
          {editing && <label className="check-row"><input type="checkbox" checked={draft.apiKey === ""} disabled={busy} onChange={event => setDraft({ ...draft, apiKey: event.target.checked ? "" : undefined })} />清除已保存的密钥，改用环境变量</label>}
          <div className="model-list-heading"><strong>模型</strong><button type="button" disabled={busy} onClick={() => setModelEdit({ model: { id: "" }, index: -1 })}>＋ 添加模型</button></div>
          {!draft.models.length && <p className="muted">可先保存供应商，再添加模型。</p>}
          {draft.models.map((model, index) => <div className="provider-model-row" key={model.id}><span><strong>{model.name || model.id}</strong><small>{model.id}</small></span><button type="button" disabled={busy} onClick={() => setModelEdit({ model, index })}>编辑</button><button type="button" className="text-danger" disabled={busy} onClick={() => setDraft({ ...draft, models: draft.models.filter((_, i) => i !== index) })}>移除</button></div>)}
        </div>
        <footer className="settings-modal-footer"><button type="button" className="reset-form" disabled={busy} onClick={() => setDraft(initial)}>重置表单</button><button type="button" disabled={busy} onClick={onClose}>取消</button><button className="primary" disabled={busy}>{busy ? "保存中…" : "保存"}</button></footer>
      </form>
    </SettingsModal>}
    {modelEdit && <ModelEditor model={modelEdit.model} editing={modelEdit.index >= 0} busy={false} error="" onClose={() => setModelEdit(null)} onSave={async model => {
      if (draft.models.some((item, index) => index !== modelEdit.index && item.id === model.id)) throw new Error("该模型 ID 已存在。");
      setDraft({ ...draft, models: modelEdit.index < 0 ? [...draft.models, model] : draft.models.map((item, index) => index === modelEdit.index ? model : item) });
      setModelEdit(null);
    }} />}
  </>;
}

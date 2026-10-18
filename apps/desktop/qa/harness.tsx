import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../src/styles.css";
import App from "../src/App";
import { scenario } from "./tauri-stub";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    {new URLSearchParams(location.search).has("parallel") && <div style={{ position: "fixed", right: 8, top: 8, zIndex: 1000, display: "flex", gap: 4 }}>
      <button onClick={() => scenario("approvals")}>测试：双会话审批</button>
      <button onClick={() => scenario("plan")}>测试：A计划</button>
      <button onClick={() => scenario("finish-a")}>测试：A后台结束</button>
      <button onClick={() => scenario("thinking")}>测试：实时思考</button>
    </div>}
  </StrictMode>,
);

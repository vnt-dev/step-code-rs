import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-only harness: serves the real src/App.tsx with the Tauri command and
// event surface aliased to qa/tauri-stub.ts, so the UI can be exercised in a
// plain browser. The production build (vite.config.ts) never touches qa/.
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [react()],
  resolve: {
    alias: [
      { find: "@tauri-apps/api/core", replacement: fileURLToPath(new URL("./qa/tauri-stub.ts", import.meta.url)) },
      { find: "@tauri-apps/api/event", replacement: fileURLToPath(new URL("./qa/tauri-stub.ts", import.meta.url)) },
    ],
  },
  server: { port: 4173, strictPort: true },
});

// Regenerate desktop icons from the transparent high-resolution master.
// Run from apps/desktop: npm run icons
import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = join(desktop, "src-tauri", "icons");
const source = join(outDir, "app-icon.svg");
const generated = mkdtempSync(join(tmpdir(), "step-code-icons-"));

try {
  // Tauri generates multi-resolution Windows ICO, macOS ICNS and window PNGs.
  execFileSync(process.execPath, [
    join(desktop, "node_modules", "@tauri-apps", "cli", "tauri.js"),
    "icon", source, "--output", generated,
  ], { cwd: desktop, stdio: "inherit" });
  for (const name of ["32x32.png", "128x128.png", "128x128@2x.png", "icon.ico", "icon.icns"]) {
    copyFileSync(join(generated, name), join(outDir, name));
  }
  execFileSync(process.execPath, [
    join(desktop, "node_modules", "@tauri-apps", "cli", "tauri.js"),
    "icon", source, "--output", generated, "--png", "1024",
  ], { cwd: desktop, stdio: "inherit" });
  copyFileSync(join(generated, "1024x1024.png"), join(outDir, "app-icon.png"));
  const publicDir = join(desktop, "public");
  mkdirSync(publicDir, { recursive: true });
  copyFileSync(join(outDir, "128x128.png"), join(publicDir, "app-icon.png"));
  console.log("Desktop and browser icons updated from", source);
} finally {
  if (dirname(resolve(generated)) === resolve(tmpdir()) && generated.startsWith(join(tmpdir(), "step-code-icons-"))) {
    rmSync(generated, { recursive: true, force: true });
  }
}

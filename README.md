# Step Code RS

**Step Code RS** 是阶跃星辰 [Step-Code](https://github.com/stepfun-ai/Step-Code) 项目的 Rust 实现版本，使用 Rust 实现模型接入、Agent 执行与会话管理，并通过 **Tauri 2 + React 19** 提供桌面客户端。你可以围绕本地项目与 AI 协作，完成代码阅读、文件修改和命令执行，并在界面中管理对话、确认工具操作、审阅执行计划及查看模型用量。

## 客户端界面

### 工作区

按项目管理聊天与历史记录，在工作区中发起编程任务。支持切换模型、调整思考级别和权限模式，以及粘贴或拖入图片作为任务上下文。

![Step Code 客户端工作区](docs/images/desktop-workspace.jpg)

### 模型设置

统一管理模型供应商、API 地址、密钥与可用模型，也可通过 Step 账户登录使用内置模型服务。

![Step Code 模型设置](docs/images/desktop-model-settings.jpg)

### 使用统计

通过活跃度热力图、用量趋势和模型分布了解使用情况，结合输出速度、首字 Token 耗时与请求明细查看模型表现。

![Step Code 使用统计](docs/images/desktop-usage.jpg)

## 主要功能

- **项目与会话管理**：按项目分组，搜索、重命名、删除会话，自动生成会话名称，并通过 JSONL 持久化历史记录。
- **编程工具**：文件读取、写入、编辑、目录浏览、内容搜索，以及 Bash / PowerShell 命令执行。
- **流式对话**：展示模型输出、思考过程、工具调用和执行结果，支持中止任务、执行中追加指令与排队跟进。
- **权限与计划审阅**：工具审批以内嵌卡片呈现，支持审阅计划，不同会话可独立等待确认。
- **多模型接入**：内置 Step 登录，支持配置 OpenAI Completions、OpenAI Responses 和 Anthropic Messages 等接口协议。
- **上下文管理**：显示上下文占用，支持自动压缩、项目指令、Skills 和提示词模板。
- **联网搜索**：内置 `search_web` 工具，支持独立搜索密钥或已保存的 Step 登录凭据。
- **用量与性能统计**：展示 Token 消耗、输出速度、首字 Token 耗时和请求记录。

## 快速开始

### 环境准备

- Rust 稳定版工具链，需支持 Rust 2024 Edition。
- Node.js：`20.19+` 或 `22.12+`，以及 npm（前端使用 Vite 7）。
- Tauri 2 对应平台的原生构建依赖。Windows 需要 Microsoft C++ Build Tools（包含 C++ 桌面开发工具与 Windows SDK）及 WebView2 Runtime；macOS 需要 Xcode Command Line Tools；Linux 需要 WebKitGTK 等系统依赖。

### 启动桌面客户端

在项目根目录执行：

```sh
cd apps/desktop
npm ci
npm run tauri dev
```

该命令会启动 Vite 开发服务并编译、打开 Tauri 桌面窗口。首次编译需要下载 Rust 依赖，耗时会较长。修改 `crates/` 下的运行时代码也会触发桌面后端重新构建。

### 首次使用

1. 打开左下角 **设置 → Step 账户**，在浏览器中完成登录；或进入 **模型设置 → 添加供应商**，填写接口地址、密钥和模型。
2. 返回工作区，添加或选择本地项目文件夹。
3. 新建聊天，选择模型、思考级别与权限模式，然后输入任务。
4. 任务执行中可查看工具记录，并按需确认操作、审阅计划或停止执行。

也可以在启动前通过环境变量提供 Step 模型密钥，例如 PowerShell：

```powershell
$env:STEP_API_KEY = "你的模型 API Key"
npm run tauri dev
```

### 仅预览 UI

如果只想查看客户端界面，无需启动 Rust 后端或配置模型密钥：

```sh
cd apps/desktop
npm ci
npx vite --config vite.harness.config.ts
```

打开 <http://localhost:4173/qa/harness.html>。该入口使用模拟 Tauri 命令与事件，仅用于界面预览和交互测试，不能实际执行编程任务。加上 `?parallel` 可显示多会话审批、计划审阅和实时思考等场景测试按钮。

## 模型与数据配置

默认数据目录为用户主目录下的 `~/.step-rs/`，可通过 `STEP_RS_HOME` 指定其他位置。项目级设置保存在 `<项目目录>/.step-rs/settings.json`。

```text
~/.step-rs/
├── settings.json       全局设置
├── auth.json           模型服务凭据
├── models.json         供应商与模型配置
├── models-store.json   远端模型目录缓存
├── trust.json          项目信任记录
├── sessions/           按项目存储的 JSONL 会话
├── skills/             用户级 Skills
├── prompts/            用户级提示词模板
└── logs/               滚动日志
```

优先通过客户端设置管理供应商。需要手动配置时，可在 `models.json` 中声明兼容接口：

```json
{
  "providers": {
    "my-provider": {
      "baseUrl": "https://api.example.com/v1",
      "api": "openai-completions",
      "models": [
        { "id": "my-model", "contextWindow": 128000, "maxTokens": 8192 }
      ]
    }
  }
}
```

上例中的地址和模型 ID 需要替换为实际服务信息。密钥可通过 `MY_PROVIDER_API_KEY` 环境变量或客户端设置提供。

`search_web` 优先使用 `STEPCODE_SEARCH_API_KEY`，否则使用 `auth.json` 中保存的 Step 凭据；`STEP_API_KEY` 仅用于模型网关，不用于搜索。若设置了非空的 `defaultTools` 白名单，需要将 `search_web` 加入其中。

## 项目结构

```text
step-code-rs/
├── Cargo.toml          Rust 工作区定义
├── crates/
│   ├── telemetry/      类型化遥测、Span 与内存记录
│   ├── providers/      模型协议、鉴权、流式请求与重试
│   ├── agent-core/     Agent 循环、工具、会话存储与上下文压缩
│   └── coding-agent/   编程会话编排、配置、权限、计划与资源管理
├── apps/desktop/
│   ├── src/            React + TypeScript 客户端
│   ├── src-tauri/      Tauri 入口、Rust 桥接、日志与请求记录
│   └── qa/             浏览器 UI 测试入口与模拟数据
└── docs/images/        README 客户端截图
```

桌面端独立于根 Rust 工作区构建，避免 Tauri 构建脚本生成的链接搜索路径影响其他 crate 的测试；两者共享仓库根目录的 `target/`。

## 开发与构建

在项目根目录检查 Rust 运行时：

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
```

在 `apps/desktop` 下检查前端与桌面后端：

```sh
npm test
npm run build
cargo test --manifest-path src-tauri/Cargo.toml --lib
```

生成桌面应用安装包：

```sh
cd apps/desktop
npm run tauri build
```

构建产物位于根目录 `target/release/`，安装包通常位于 `target/release/bundle/`。`npm run build` 只构建前端资源，生成桌面应用需要运行 Tauri 构建命令。

日志默认写入 `~/.step-rs/logs/step-code.log`，可通过 `RUST_LOG` 调整级别。更多实现说明见 [桌面端文档](apps/desktop/README.md)、[Agent Core](crates/agent-core/README.md)、[Providers](crates/providers/README.md) 和 [Telemetry](crates/telemetry/README.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。

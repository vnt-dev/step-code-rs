# step-harness-telemetry

面向 Rust 的显式 telemetry 抽象。提供父子 span、属性与事件、状态、
空实现、内存快照、可序列化 schema，以及适配器一致性检查。
功能范围参考 Step-Code 的 telemetry 包；API 使用 Rust 的类型与生命周期，
不要求与 TypeScript API 或 JSON 数值类型保持兼容。

## 普通操作与可失败操作

```rust
use step_harness_telemetry::{
    InMemoryTelemetryContext, SpanAttributes, SpanOptions,
    TelemetryContext, TelemetrySpan,
};

async fn example() -> Result<u64, std::io::Error> {
    let context = InMemoryTelemetryContext::new();
    context.start_result_span(
        SpanOptions::new("account.load").with_attribute("account.id", u64::MAX),
        |span| async move {
            span.add_event("cache.lookup", SpanAttributes::new());
            let count = span.start_span(
                SpanOptions::new("account.read"), |_| async { 3_u64 }
            ).await;
            span.set_attributes([("account.count".into(), count.into())].into());
            Ok(count)
        },
    ).await
}
```

- `start_span` 原样返回任意值；不会根据 `bool`、`Option` 或其他业务数据猜测成功与否。
- `start_result_span` 原样返回 `Result<T, E>`，对 `Err` 自动记录错误类型名和
  `Display` 文本；错误无需实现 `Clone`、`std::error::Error` 或 `'static`。
- 显式 `set_status` 后，以最后一次显式状态为准，自动错误和取消状态不覆盖它。
- callback 在返回的 Future 首次被 poll 时调用一次。未 poll 就丢弃的 Future 不创建 span。
- 操作完成时结束 span；已开始的 Future 被丢弃时，RAII guard 结束 span，
  没有显式状态时记为 `Cancelled`。取消原因不会被进一步分类。
- span handle 可以克隆，但克隆不延长操作的生命周期。已结束后的记录方法无效，
  后续子操作仍执行并保留返回值，但不记录。
- 不捕获 panic，不使用 panic 表达业务失败，不依赖 unwind payload。
  guard 只负责正常析构时收尾；进程 abort 时不承诺清理。
- trait 使用静态分发；context 可借用应用状态。需要跨线程执行的 Future 为 `Send`。
- 错误文本会进入诊断记录，业务可以使用适当的错误类型或显式状态控制这些内容。

## 数值与属性

`AttributeValue` 明确区分以下类型：

| 类型                                     | Rust 数据                  |
|----------------------------------------|--------------------------|
| `Integer` / `Integers`                 | `i64` / `Vec<i64>`       |
| `UnsignedInteger` / `UnsignedIntegers` | `u64` / `Vec<u64>`       |
| `Float` / `Floats`                     | `f64` / `Vec<f64>`       |
| `String` / `Strings`                   | `String` / `Vec<String>` |
| `Boolean` / `Booleans`                 | `bool` / `Vec<bool>`     |

整数转换不会经过浮点数，`i64::MIN`、`i64::MAX`、`u64::MAX` 均精确保留。
常用的 `i8/i16/i32/u8/u16/u32` 可以无损转换；`f32` 无损扩展为 `f64`。
`usize/isize/i128/u128` 需要调用方自行做检查转换。
仅实际浮点测量使用 `Float`；JSON 序列化时使用有限浮点值。

`SpanAttributes` 是 `BTreeMap<String, AttributeValue>`，后写入的同名属性覆盖之前的值。
事件按记录顺序保留。属性值采用带标签的 Serde 表示，空数组不会丢失类型：

```json
{
  "type": "unsigned_integer",
  "value": 18446744073709551615
}
```

## 编译期 span 契约

`SpanDefinition` 用关联类型绑定一个 span 的名称、起始属性、结束属性与事件。
必填字段使用 struct 字段，可选字段使用 `Option`，闭集值和事件使用 enum。
不同 span 使用不同 marker type；不再传入实际未使用的 schema 参数。

```rust
use std::convert::Infallible;
use step_harness_telemetry::*;

struct Load;
struct LoadInput {
    account_id: u64
}
struct LoadOutput {
    found: bool
}

impl From<LoadInput> for SpanAttributes {
    fn from(input: LoadInput) -> Self {
        [("account.id".into(), input.account_id.into())].into()
    }
}
impl From<LoadOutput> for SpanAttributes {
    fn from(output: LoadOutput) -> Self {
        [("account.found".into(), output.found.into())].into()
    }
}
impl SpanDefinition for Load {
    const NAME: &'static str = "account.load";
    type StartAttributes = LoadInput;
    type EndAttributes = LoadOutput;
    type Event = Infallible; // 该 span 没有事件
}

async fn example() {
    let context = InMemoryTelemetryContext::new();
    let typed = TypedTelemetryContext::new(&context);
    typed.start_span::<Load, _, _, _>(
        LoadInput { account_id: 42 },
        |span| async move {
            span.set_attributes(LoadOutput { found: true });
        },
    ).await;
}
```

callback 获得 `TypedSpan`，只能记录该 span 指定的结束属性和事件类型。
使用 `span.children()` 创建绑定到当前 span 的 typed child context。
没有属性时可用 `NoAttributes`；使用开放 `SpanAttributes` 则表示有意允许任意属性。
领域负责实现属性映射和为 span 选择唯一名称；本 crate 不生成领域代码。

## Schema 元数据

`TelemetrySchemaDefinition` 保留版本、span 描述、父规则、起始/结束属性、
事件、敏感性、基数和状态规则等可序列化元数据。它与编译期契约相互独立，
不对运行时 recording 做校验，也不自动从 JSON 生成 Rust 类型。

`TelemetryAttributeDefinition` 是以 `type` 为标签的枚举，
数值类型明确使用 `integer`、`unsigned_integer`、`float` 及对应的数组类型。
`values`、`elementValues` 和 `examples` 的 Rust 数据类型由变体决定：

```json
{
  "type": "integer[]",
  "description": "Observed counts",
  "elementValues": [],
  "examples": [
    []
  ]
}
```

空数组和空嵌套数组能精确往返；整数定义不接受浮点值或字符串值。
敏感性、基数、允许值、父规则和 `errorWhen` 是描述性元数据；
其业务约束应由领域类型或应用校验实现。

## 空实现与内存实现

`NOOP_TELEMETRY_CONTEXT` / `NoopTelemetryContext` 保留 callback 的值和错误，
不记录数据，也不格式化错误文本。嵌套 span 使用无状态的 `NoopSpan`。

`InMemoryTelemetryContext` 可以克隆以共享一个记录作用域。
`get_spans()` 返回原子、独立的快照，包含父 ID、属性、事件、状态、
结束标记及结束序号；修改快照不影响内部数据。父结束检查和子 span admission
在同一锁内完成；所有同步锁均不跨越 `.await`。

该实现用于测试或有界的本地诊断，存储不设容量上限，没有时间戳或 exporter。
生产后端适配器自行负责时间、采样、导出与 flush。

## 适配器检查

实现 `TelemetryContext`、`TelemetrySpan` 后，用
`testing::check_adapter_conformance` 验证 callback、Result、记录合并、
显式状态、取消、已结束 handle 和真实重叠执行的子 span。

fixture 的 `get_spans()` 需要包含 open span，并在必要时 flush exporter。
每组检查使用独立 fixture；测试 runner 负责处理意外 panic。

## 实现依赖与迁移

- `parking_lot`：共享记录状态的同步锁。
- `futures-util` / `futures-channel`：适配器检查中的 Future 组合和 oneshot 协调。
- `serde`：属性与 schema 的序列化。
- demo 使用 `futures::executor::block_on`，没有手写 executor。
- crate 和 demo 都声明 `#![forbid(unsafe_code)]`。

本次 API 调整需要迁移调用方：

- `Number/Numbers` 替换为明确的整数或浮点变体，schema 的 `number` 也需明确类型。
- `SpanOptions.attributes` 与事件属性直接使用 map；空 map 替代 `None`。
- span 本身实现 `TelemetryContext`，无需重复一套 child starter 方法。
- `create_typed_span_starter`、identity schema helper、panic helper 和 boxed case runner 已移除。
- typed 调用使用 `TypedTelemetryContext` 和 `SpanDefinition`；适配器测试使用
  `check_adapter_conformance`。

## 检查

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run
```

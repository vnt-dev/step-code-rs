//! Harness layer: execution-environment contracts, built-in tools, durable
//! sessions, and context compaction.

pub mod agent_harness;
pub mod env;
pub mod events;
mod frontmatter;
pub mod messages;
pub mod prompt_templates;
pub mod reducer;
pub mod skill_discovery;
pub mod skills;
pub mod system_prompt;
pub mod telemetry;
pub(crate) mod timeout;
pub mod tools;
pub mod types;
pub mod utils;

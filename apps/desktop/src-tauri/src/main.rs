//! Tauri application entry point.
//!
//! The desktop shell is a thin layer over the `step-harness-coding-agent`
//! runtime: `lib.rs` owns the agent runtime state, `main.rs` only boots it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    step_code_desktop_lib::run()
}

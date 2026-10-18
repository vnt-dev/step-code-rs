//! Tauri application library: wires the coding-agent runtime to the web UI.

mod bridge;
mod external_links;
mod logging;
mod usage;
mod request_store;

use step_harness_coding_agent::AppPaths;

fn init_logging() {
    // Logging is best effort: the app runs without a log file.
    logging::init(&AppPaths::discover().logs_dir()).ok();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_logging();
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            external_links::open_external_url,
            bridge::get_bootstrap,
            bridge::new_session,
            bridge::add_project,
            bridge::remove_project,
            bridge::open_session,
            bridge::delete_session,
            bridge::list_sessions,
            bridge::get_transcript,
            bridge::recall_last_user_message,
            bridge::prompt,
            bridge::list_skills,
            bridge::abort,
            bridge::steer,
            bridge::follow_up,
            bridge::set_model,
            bridge::set_thinking_level,
            bridge::set_approval_mode,
            bridge::respond_approval,
            bridge::enter_plan_mode,
            bridge::exit_plan_mode,
            bridge::toggle_plan_mode,
            bridge::respond_plan_review,
            bridge::compact,
            bridge::update_settings,
            bridge::rename_session,
            bridge::set_session_project,
            bridge::refresh_models,
            bridge::reload,
            bridge::get_stats,
            bridge::get_usage_statistics,
            bridge::get_model_health,
            bridge::list_request_records,
            bridge::get_providers,
            bridge::save_provider,
            bridge::delete_provider,
            bridge::login_step,
            bridge::cancel_step_login,
            bridge::reopen_step_login,
            bridge::logout_step,
        ])
        .setup(|app| {
            bridge::setup(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Step Code")
        .run(|app, event| { if matches!(event, tauri::RunEvent::Exit) { use tauri::Manager; app.state::<bridge::AppState>().requests.shutdown(); } });
}

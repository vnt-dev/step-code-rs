//! User-Agent string for provider requests.

/// `stepcode (<platform> <release>; <arch>)`, matching the TypeScript product
/// identifier.
pub fn get_user_agent() -> String {
    format!(
        "stepcode ({} {}; {})",
        std::env::consts::OS,
        os_release(),
        std::env::consts::ARCH
    )
}

fn os_release() -> String {
    std::process::Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

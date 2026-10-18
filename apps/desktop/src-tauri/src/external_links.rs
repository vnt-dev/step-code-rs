//! Open message links outside the application webview.

#[tauri::command]
pub fn open_external_url(url: String) -> Result<(), String> {
    let url = browser_url(&url)?;
    open::that(url.as_str()).map_err(|error| format!("无法打开浏览器：{error}"))
}

fn browser_url(value: &str) -> Result<tauri::Url, String> {
    let url = tauri::Url::parse(value).map_err(|_| "链接地址无效。".to_owned())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("仅支持在浏览器中打开 HTTP 或 HTTPS 链接。".to_owned());
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_web_links_and_rejects_local_paths_and_other_protocols() {
        for value in [
            "https://example.com/path?q=a&b=2#section",
            "http://localhost:3000",
        ] {
            assert!(browser_url(value).is_ok(), "{value}");
        }
        for value in [
            "/project/README.md",
            "#section",
            "file:///C:/test.txt",
            "javascript:alert(1)",
            "data:text/html,test",
            "ms-settings:display",
        ] {
            assert!(browser_url(value).is_err(), "{value}");
        }
    }
}

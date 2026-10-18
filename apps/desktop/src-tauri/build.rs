fn main() {
    // Windows icons are compiled into the executable by tauri-build. Track
    // their inputs explicitly so each Cargo feature/profile rebuilds its
    // resource cache after an icon change (including `tauri dev`).
    println!("cargo:rerun-if-changed=icons");
    tauri_build::build()
}

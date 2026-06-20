fn main() {
    #[cfg(not(windows))]
    tauri_build::build();

    #[cfg(windows)]
    {
        std::env::set_var("TAURI_WINRES_DISABLED", "1");
        tauri_build::build();
    }
}

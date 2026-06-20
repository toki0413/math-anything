#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{Manager, State, AppHandle, Emitter};
use tauri_plugin_shell::ShellExt;

struct Backend {
    port: Mutex<u16>,
    child: Mutex<Option<tauri_plugin_shell::process::CommandChild>>,
    restart_count: Mutex<u32>,
    max_restarts: u32,
}

#[tauri::command]
async fn get_backend_port(state: State<'_, Backend>) -> Result<u16, String> {
    let port = state.port.lock().map_err(|e| e.to_string())?;
    Ok(*port)
}

#[tauri::command]
fn get_extraction_engines() -> Vec<String> {
    vec![
        "vasp".to_string(),
        "lammps".to_string(),
        "abaqus".to_string(),
        "ansys".to_string(),
        "gaussian".to_string(),
        "cp2k".to_string(),
        "quantum_espresso".to_string(),
        "openfoam".to_string(),
        "su2".to_string(),
        "dakota".to_string(),
        "fluent".to_string(),
        "comsol".to_string(),
        "nwchem".to_string(),
        "gamess".to_string(),
        "fenics".to_string(),
        "elmerfem".to_string(),
        "dealii".to_string(),
        "mfem".to_string(),
    ]
}

#[tauri::command]
fn check_rust_acceleration() -> bool {
    // We're running inside the Rust process, so the module is available
    true
}

#[tauri::command]
fn rust_health_check() -> serde_json::Value {
    serde_json::json!({
        "status": "ok",
        "rust_available": true,
        "version": env!("CARGO_PKG_VERSION"),
        "num_cpus": num_cpus::get(),
    })
}

fn find_free_port() -> u16 {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
    listener.local_addr().unwrap().port()
}

fn wait_for_server(port: u16, timeout_secs: u64) -> bool {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    while start.elapsed() < timeout {
        if std::net::TcpStream::connect(format!("127.0.0.1:{}", port)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn check_health(port: u16) -> bool {
    std::net::TcpStream::connect(format!("127.0.0.1:{}", port)).is_ok()
}

fn start_backend(app: &tauri::App, port: u16) -> Result<tauri_plugin_shell::process::CommandChild, String> {
    let sidecar = app
        .shell()
        .sidecar("math-anything-server")
        .map_err(|e| format!("Sidecar not found: {}. Run build_sidecar.py first.", e))?;

    let (_rx, child) = sidecar
        .args([&port.to_string()])
        .spawn()
        .map_err(|e| format!("Failed to start backend: {}", e))?;

    if !wait_for_server(port, 30) {
        return Err("Backend did not start within 30 seconds".into());
    }

    Ok(child)
}

fn spawn_health_monitor(app_handle: AppHandle, state: Arc<Backend>) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(5));
            let port = match state.port.lock() {
                Ok(p) => *p,
                Err(_) => continue,
            };
            if !check_health(port) {
                let should_restart = match state.restart_count.lock() {
                    Ok(mut count) => {
                        if *count < state.max_restarts {
                            *count += 1;
                            true
                        } else {
                            false
                        }
                    }
                    Err(_) => false,
                };
                if should_restart {
                    if let Ok(mut child_guard) = state.child.lock() {
                        if let Some(old_child) = child_guard.take() {
                            let _ = old_child.kill();
                        }
                    }
                    let sidecar = app_handle.shell()
                        .sidecar("math-anything-server");
                    if let Ok(cmd) = sidecar {
                        if let Ok((_rx, new_child)) = cmd.args([&port.to_string()]).spawn() {
                            if let Ok(mut child_guard) = state.child.lock() {
                                *child_guard = Some(new_child);
                            }
                            let _ = app_handle.emit("backend-restarted", ());
                        }
                    }
                } else {
                    let _ = app_handle.emit("backend-crashed", ());
                }
            }
        }
    });
}

fn main() {
    let port = find_free_port();
    let port_for_setup = port;

    let backend_state = Arc::new(Backend {
        port: Mutex::new(port),
        child: Mutex::new(None),
        restart_count: Mutex::new(0),
        max_restarts: 3,
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(backend_state.clone())
        .invoke_handler(tauri::generate_handler![
            get_backend_port,
            get_extraction_engines,
            check_rust_acceleration,
            rust_health_check
        ])
        .setup(move |app| {
            let port = port_for_setup;

            match start_backend(app, port) {
                Ok(child) => {
                    *backend_state.child.lock().unwrap() = Some(child);
                    println!("Backend started on port {}", port);
                }
                Err(e) => {
                    eprintln!("Warning: Could not start backend sidecar: {}", e);
                    eprintln!("The app will try to connect to http://localhost:8000 instead.");
                    *backend_state.port.lock().unwrap() = 8000;
                }
            }

            let app_handle = app.handle().clone();
            spawn_health_monitor(app_handle, backend_state.clone());

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state: State<Backend> = window.state();
                if let Ok(mut child_guard) = state.child.lock() {
                    if let Some(child) = child_guard.take() {
                        let _ = child.kill();
                    }
                };
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

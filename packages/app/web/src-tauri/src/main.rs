// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::process::{Command, Stdio};
use std::sync::OnceLock;

static ARTIFACT_DIR: OnceLock<String> = OnceLock::new();
static PYTHON_BIN: OnceLock<String> = OnceLock::new();

fn python_command() -> Command {
    let interpreter = PYTHON_BIN.get().map(String::as_str).unwrap_or("python3");
    Command::new(interpreter)
}

fn artifact_dir() -> &'static str {
    ARTIFACT_DIR.get().map(String::as_str).unwrap_or("packages/models/artifacts/linguae")
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Linguae.", name)
}

#[tauri::command]
fn detect(text: String, top_k: Option<u32>) -> String {
    let k = top_k.unwrap_or(3).clamp(1, 10);
    let payload = serde_json::json!({
        "text": text,
        "top_k": k,
    });

    let bridge = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("tauri_bridge.py")))
        .unwrap_or_else(|| std::path::PathBuf::from("tauri_bridge.py"));

    let mut child = python_command()
        .arg(&bridge)
        .env("LINGUAE_ARTIFACT_DIR", artifact_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn python bridge");

    use std::io::Write;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(payload.to_string().as_bytes());
    }

    let output = child.wait_with_output().expect("python bridge did not run");
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return serde_json::json!({ "error": err.trim() }).to_string();
    }

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let artifact_dir = std::env::var("LINGUAE_ARTIFACT_DIR")
        .unwrap_or_else(|_| "packages/models/artifacts/linguae".to_string());
    let _ = ARTIFACT_DIR.set(artifact_dir);

    let python_bin = std::env::var("LINGUAE_PYTHON_BIN")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());
    let _ = PYTHON_BIN.set(python_bin);

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, detect])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
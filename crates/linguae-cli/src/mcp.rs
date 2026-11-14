//! MCP server over stdio JSON-RPC.
//!
//! Tools delegate to the `linguae` binary's own CLI so MCP behaviour always
//! matches the command line. Configure an MCP client with: `linguae mcp`

use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::process::Command;

const SERVER_INFO: &str = "linguae-mcp";
const PROTOCOL_VERSION: &str = "2024-11-05";

fn arg(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|s| !s.is_empty())
}

fn run_cli(args: &[String]) -> Value {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(err) => {
            return json!({
                "content": [{"type": "text", "text": format!("cannot locate linguae binary: {err}")}],
                "isError": true
            })
        }
    };
    match Command::new(exe).args(args).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if output.status.success() {
                let text = if stdout.is_empty() { stderr } else { stdout };
                json!({"content": [{"type": "text", "text": text}]})
            } else {
                json!({
                    "content": [{"type": "text", "text": stderr}],
                    "isError": true
                })
            }
        }
        Err(err) => json!({
            "content": [{"type": "text", "text": format!("failed to run linguae: {err}")}],
            "isError": true
        }),
    }
}

fn build_args(name: &str, params: &Value) -> Option<Vec<String>> {
    match name {
        "detect_language" => {
            let text = arg(params, "text");
            let file = arg(params, "file");
            if text.is_none() && file.is_none() {
                return None;
            }
            let mut args = Vec::new();
            if let Some(file) = file {
                args.extend(["--file".to_string(), file]);
            } else if let Some(text) = text {
                args.push(text);
            }
            if let Some(v) = params.get("top_k").and_then(Value::as_u64) {
                args.extend(["--top-k".to_string(), v.to_string()]);
            }
            if let Some(v) = arg(params, "device") {
                args.extend(["--device".to_string(), v]);
            }
            if let Some(v) = arg(params, "artifact") {
                args.extend(["--artifact".to_string(), v]);
            }
            args.push("--json".to_string());
            Some(args)
        }
        _ => None,
    }
}

fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "detect_language",
                "description": "Detect the language of text (ISO 639-3 code) using the local Linguae model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to classify"},
                        "file": {"type": "string", "description": "Classify each non-empty line of a UTF-8 file"},
                        "top_k": {"type": "integer", "description": "Number of alternatives in JSON output (1-20, default 3)"},
                        "device": {"type": "string", "description": "Inference device: cpu, cuda or metal"},
                        "artifact": {"type": "string", "description": "Artifact directory (default: crates/linguae-train/artifacts/linguae)"}
                    },
                    "additionalProperties": false
                }
            }
        ]
    })
}

fn handle_request(request: &Value) -> Option<Value> {
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let has_id = request.get("id").is_some();

    let result = match method {
        "initialize" => json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_INFO, "version": env!("CARGO_PKG_VERSION")}
        }),
        "ping" => Value::Null,
        "tools/list" => tools_list(),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or(Value::Null);
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            match build_args(name, &arguments) {
                Some(args) => run_cli(&args),
                None => json!({
                    "content": [{"type": "text", "text": format!("Unknown tool or missing required arguments: {name}")}],
                    "isError": true
                }),
            }
        }
        _ => {
            if !has_id {
                return None;
            }
            return Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("Method not found: {method}")}
            }));
        }
    };

    if !has_id {
        return None;
    }
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// Run the MCP server on stdin/stdout until EOF.
pub fn run() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(request) => {
                if let Some(response) = handle_request(&request) {
                    writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                    stdout.flush()?;
                }
            }
            Err(err) => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {"code": -32700, "message": format!("Parse error: {err}")}
                });
                writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                stdout.flush()?;
            }
        }
    }
    Ok(())
}

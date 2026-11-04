//! Integration tests for the `linguae` binary.

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;

const ENGLISH: &str = "Hello world, this is a longer English text to test language detection";
const FRENCH: &str = "Bonjour tout le monde, comment allez-vous aujourd'hui dans ce beau monde";

fn artifact_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/models/artifacts/linguae")
        .canonicalize()
        .expect("the repository artifact must exist")
}

fn linguae() -> Command {
    let mut command = Command::cargo_bin("linguae").expect("binary is built");
    command.env("LINGUAE_ARTIFACT_DIR", artifact_directory());
    command
}

#[test]
fn prints_version() {
    linguae()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::starts_with("linguae "));
}

#[test]
fn prints_help() {
    linguae()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Detect the language"))
        .stdout(predicate::str::contains("--artifact"))
        .stdout(predicate::str::contains("--confidence"))
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn classifies_a_single_text() {
    linguae().arg(ENGLISH).assert().success().stdout("eng\n");
}

#[test]
fn prints_confidence() {
    linguae()
        .args(["--confidence", ENGLISH])
        .assert()
        .success()
        .stdout(predicate::str::is_match(r"^eng\t0\.\d{4}\n$").unwrap());
}

#[test]
fn emits_json_lines() {
    let output = linguae()
        .args(["--json", ENGLISH])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let value: serde_json::Value = serde_json::from_slice(&output).expect("valid JSON");
    assert_eq!(value["text"], ENGLISH);
    assert_eq!(value["language"], "eng");
    assert!(value["confidence"].as_f64().unwrap() > 0.5);
    assert_eq!(value["top_k"].as_array().unwrap().len(), 3);
}

#[test]
fn json_preserves_non_ascii_text() {
    let output = linguae()
        .args(["--json", FRENCH])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("UTF-8 output");
    assert!(text.contains("Bonjour tout le monde"));
    assert!(!text.contains("\\u"));
}

#[test]
fn honours_top_k() {
    let output = linguae()
        .args(["--json", "--top-k", "5", ENGLISH])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let value: serde_json::Value = serde_json::from_slice(&output).expect("valid JSON");
    assert_eq!(value["top_k"].as_array().unwrap().len(), 5);
}

#[test]
fn rejects_top_k_outside_the_allowed_range() {
    linguae().args(["--top-k", "0", ENGLISH]).assert().code(2);
    linguae().args(["--top-k", "21", ENGLISH]).assert().code(2);
}

#[test]
fn classifies_each_line_of_a_file() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("sentences.txt");
    std::fs::write(&path, format!("{ENGLISH}\n\n{FRENCH}\n")).expect("write file");

    linguae()
        .args(["--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout("eng\nfra\n");
}

#[test]
fn reads_stdin_when_text_is_omitted() {
    linguae()
        .write_stdin(format!("{ENGLISH}\n"))
        .assert()
        .success()
        .stdout("eng\n");
}

#[test]
fn rejects_text_and_file_together() {
    linguae()
        .args([ENGLISH, "--file", "anything.txt"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "provide either TEXT or --file, not both",
        ));
}

#[test]
fn rejects_empty_text() {
    linguae()
        .arg("   ")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("text is empty"));
}

#[test]
fn rejects_empty_stdin() {
    linguae()
        .write_stdin("")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no input text received on stdin"));
}

#[test]
fn rejects_a_missing_file() {
    linguae()
        .args(["--file", "/definitely/not/a/real/file.txt"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("could not read"));
}

#[test]
fn rejects_a_missing_artifact_directory() {
    linguae()
        .args(["--artifact", "/definitely/not/a/real/artifact", ENGLISH])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("artifact directory not found"));
}

#[test]
fn rejects_an_unknown_device() {
    linguae()
        .args(["--device", "tpu", ENGLISH])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("unsupported device"));
}

//! Integration tests for the `linguae-data` binary.

mod common;

use std::fs;

use assert_cmd::Command;
use common::Fixture;
use predicates::prelude::*;

fn linguae_data() -> Command {
    Command::cargo_bin("linguae-data").expect("binary is built")
}

#[test]
fn prints_help() {
    linguae_data()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("WiLI-2018"))
        .stdout(predicate::str::contains("wili"))
        .stdout(predicate::str::contains("tatoeba"))
        .stdout(predicate::str::contains("words"))
        .stdout(predicate::str::contains("all"));
}

#[test]
fn runs_every_conversion_with_the_all_subcommand() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("The cat", "eng"), ("le chat", "fra")]);
    fixture.write_wili_split("test", &[("the dog", "eng"), ("le chien", "fra")]);
    fixture.write_file(
        "csv/tatoeba.tsv",
        "1\teng\tHello, world\n2\tfra\tBonjour le monde\n",
    );

    linguae_data()
        .args(["--data-dir", fixture.root().to_str().unwrap()])
        .args(["--log-level", "error"])
        .args(["all", "--max-words", "10"])
        .assert()
        .success()
        .stdout(predicate::str::contains("WiLI-2018:"))
        .stdout(predicate::str::contains("Tatoeba:"))
        .stdout(predicate::str::contains("Words:"));

    let csv = fixture.csv_dir();
    assert!(csv.join("wili-2018.csv").is_file());
    assert!(csv.join("tatoeba.csv").is_file());
    assert!(csv.join("wili-2018-words.csv").is_file());
}

#[test]
fn words_train_only_skips_the_test_split() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("alpha beta", "eng")]);
    fixture.write_wili_split("test", &[("gamma delta", "eng")]);

    linguae_data()
        .args(["--data-dir", fixture.root().to_str().unwrap()])
        .args(["--log-level", "error"])
        .args(["words", "--train-only"])
        .assert()
        .success();

    let contents = fs::read_to_string(fixture.csv_dir().join("wili-2018-words.csv"))
        .expect("word output is readable");
    assert!(contents.contains("alpha,eng,1"));
    assert!(contents.contains("beta,eng,1"));
    assert!(!contents.contains("gamma"));
    assert!(!contents.contains("delta"));
}

#[test]
fn honours_explicit_input_and_output_paths() {
    let fixture = Fixture::new();
    let input = fixture.write_file("custom/tatoeba.tsv", "1\teng\tHello\n2\t\\N\tignored\n");
    let output = fixture.root().join("custom/out.csv");

    linguae_data()
        .args(["--data-dir", fixture.root().to_str().unwrap()])
        .args(["--log-level", "error"])
        .arg("tatoeba")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .assert()
        .success();

    assert_eq!(
        fs::read_to_string(&output).expect("output is readable"),
        "id,language,text\n1,eng,Hello\n"
    );
}

#[test]
fn reports_errors_with_a_non_zero_exit_code() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("Hello", "eng")]);

    linguae_data()
        .args(["--data-dir", fixture.root().to_str().unwrap()])
        .args(["--log-level", "error"])
        .arg("wili")
        .assert()
        .failure()
        .code(1)
        .stderr(predicate::str::contains("error:"))
        .stderr(predicate::str::contains("x_test.txt"));
}

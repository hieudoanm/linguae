//! Integration tests for per-language word extraction.

mod common;

use std::fs;

use common::Fixture;
use linguae_data::{extract_words, Error, WordOptions};

fn read_rows(path: &std::path::Path) -> Vec<(String, String, u64)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("output is readable");
    reader
        .records()
        .map(|record| {
            let record = record.expect("record is valid");
            (
                record.get(0).unwrap_or_default().to_string(),
                record.get(1).unwrap_or_default().to_string(),
                record
                    .get(2)
                    .unwrap_or_default()
                    .parse()
                    .expect("count is an integer"),
            )
        })
        .collect()
}

fn fixture() -> Fixture {
    let fixture = Fixture::new();
    fixture.write_wili_split(
        "train",
        &[("The cat the dog", "eng"), ("le chat le chien", "fra")],
    );
    fixture.write_wili_split("test", &[("the THE cat", "eng")]);
    fixture
}

#[test]
fn extracts_words_sorted_by_frequency_with_a_cap() {
    let fixture = fixture();
    let output = fixture.csv_dir().join("wili-2018-words.csv");

    let report = extract_words(&WordOptions {
        input_dir: fixture.wili_dir(),
        output: output.clone(),
        max_words: 2,
        include_test: true,
    })
    .expect("extraction succeeds");

    assert_eq!(report.languages, 2);
    assert_eq!(report.words, 4);
    assert_eq!(report.max_words_per_language, 2);
    assert_eq!(report.words_per_language.min, 2);
    assert_eq!(report.words_per_language.max, 2);

    assert_eq!(
        fs::read_to_string(&output).expect("output is readable"),
        "text,language,count\n\
         the,eng,4\n\
         cat,eng,2\n\
         le,fra,2\n\
         chat,fra,1\n"
    );
}

#[test]
fn train_only_excludes_the_test_split() {
    let fixture = fixture();
    let output = fixture.csv_dir().join("wili-2018-words.csv");

    extract_words(&WordOptions {
        input_dir: fixture.wili_dir(),
        output: output.clone(),
        max_words: 0,
        include_test: false,
    })
    .expect("extraction succeeds");

    let rows = read_rows(&output);
    assert_eq!(
        rows,
        vec![
            ("the".to_string(), "eng".to_string(), 2),
            ("cat".to_string(), "eng".to_string(), 1),
            ("dog".to_string(), "eng".to_string(), 1),
            ("le".to_string(), "fra".to_string(), 2),
            ("chat".to_string(), "fra".to_string(), 1),
            ("chien".to_string(), "fra".to_string(), 1),
        ]
    );
}

#[test]
fn zero_cap_keeps_every_word() {
    let fixture = fixture();
    let output = fixture.csv_dir().join("wili-2018-words.csv");

    let report = extract_words(&WordOptions {
        input_dir: fixture.wili_dir(),
        output: output.clone(),
        max_words: 0,
        include_test: true,
    })
    .expect("extraction succeeds");

    assert_eq!(report.words, 6);
    assert_eq!(report.words_per_language.min, 3);
    assert_eq!(report.words_per_language.max, 3);
    assert_eq!(read_rows(&output).len(), 6);
}

#[test]
fn rejects_a_dataset_without_words() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("2024 — 42!", "eng")]);
    fixture.write_wili_split("test", &[("2025 — 43!", "eng")]);

    let error = extract_words(&WordOptions {
        input_dir: fixture.wili_dir(),
        output: fixture.csv_dir().join("wili-2018-words.csv"),
        max_words: 0,
        include_test: true,
    })
    .expect_err("a dataset without words is rejected");

    match error {
        Error::InvalidData(message) => assert!(message.contains("no words")),
        other => panic!("unexpected error: {other}"),
    }
}

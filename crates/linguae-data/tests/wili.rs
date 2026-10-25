//! Integration tests for the WiLI-2018 conversion.

mod common;

use std::fs;

use common::Fixture;
use linguae_data::{convert_wili, Error, WiliOptions};

#[test]
fn converts_splits_in_order_and_reports_statistics() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("Hello world", "eng"), ("Un texte", "fra")]);
    fixture.write_wili_split(
        "test",
        &[("Bonjour le monde", "fra"), ("Hello world", "deu")],
    );

    let output = fixture.csv_dir().join("wili-2018.csv");
    let report = convert_wili(&WiliOptions {
        input_dir: fixture.wili_dir(),
        output: output.clone(),
    })
    .expect("conversion succeeds");

    assert_eq!(report.train_rows, 2);
    assert_eq!(report.test_rows, 2);
    assert_eq!(report.rows, 4);
    assert_eq!(report.languages, 3);
    assert_eq!(report.duplicate_text_rows, 1);
    assert_eq!(report.texts_with_conflicting_labels, 1);
    assert_eq!(report.lengths.min, 8);
    assert_eq!(report.lengths.max, 16);
    assert!(report.file.bytes > 0);
    assert_eq!(report.file.sha256.len(), 64);

    let contents = fs::read_to_string(&output).expect("output is readable");
    assert_eq!(
        contents,
        "text,language\n\
         Hello world,eng\n\
         Un texte,fra\n\
         Bonjour le monde,fra\n\
         Hello world,deu\n"
    );
}

#[test]
fn quotes_texts_that_contain_commas() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("Paris, France", "fra")]);
    fixture.write_wili_split("test", &[("Berlin, Germany", "deu")]);

    let output = fixture.csv_dir().join("wili-2018.csv");
    convert_wili(&WiliOptions {
        input_dir: fixture.wili_dir(),
        output: output.clone(),
    })
    .expect("conversion succeeds");

    let contents = fs::read_to_string(&output).expect("output is readable");
    assert_eq!(
        contents,
        "text,language\n\"Paris, France\",fra\n\"Berlin, Germany\",deu\n"
    );
}

#[test]
fn rejects_splits_with_mismatched_texts_and_labels() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[("Hello", "eng"), ("Bonjour", "fra")]);
    fixture.write_wili_split("test", &[("Hola", "spa")]);
    fs::write(fixture.wili_dir().join("y_train.txt"), "eng\n").expect("labels are written");

    let error = convert_wili(&WiliOptions {
        input_dir: fixture.wili_dir(),
        output: fixture.csv_dir().join("wili-2018.csv"),
    })
    .expect_err("mismatched splits are rejected");

    match error {
        Error::InvalidData(message) => assert!(message.contains("2 texts but 1 labels")),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rejects_an_empty_split() {
    let fixture = Fixture::new();
    fixture.write_wili_split("train", &[]);
    fixture.write_wili_split("test", &[("Hola", "spa")]);

    let error = convert_wili(&WiliOptions {
        input_dir: fixture.wili_dir(),
        output: fixture.csv_dir().join("wili-2018.csv"),
    })
    .expect_err("an empty split is rejected");

    match error {
        Error::InvalidData(message) => assert!(message.contains("train split is empty")),
        other => panic!("unexpected error: {other}"),
    }
}

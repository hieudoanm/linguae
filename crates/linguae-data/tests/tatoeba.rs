//! Integration tests for the Tatoeba TSV conversion.

mod common;

use std::fs;

use common::Fixture;
use linguae_data::{convert_tatoeba, Error, TatoebaOptions};

fn read_csv(path: &std::path::Path) -> Vec<Vec<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("output is readable");
    reader
        .records()
        .map(|record| {
            record
                .expect("record is valid")
                .iter()
                .map(str::to_string)
                .collect()
        })
        .collect()
}

#[test]
fn converts_tsv_and_drops_invalid_labels() {
    let fixture = Fixture::new();
    let input = fixture.write_file(
        "csv/tatoeba.tsv",
        "1\teng\tHello, world\n\
         2\t\\N\tignored\n\
         3\t\tblank\n\
         4\tfra\tBonjour le monde\n",
    );
    let output = fixture.csv_dir().join("tatoeba.csv");

    let report = convert_tatoeba(&TatoebaOptions {
        input,
        output: output.clone(),
    })
    .expect("conversion succeeds");

    assert_eq!(report.input_rows, 4);
    assert_eq!(report.output_rows, 2);
    assert_eq!(report.dropped_rows, 2);
    assert_eq!(report.languages, 2);
    assert_eq!(report.lengths.min, 12);
    assert_eq!(report.lengths.max, 16);
    assert_eq!(report.file.sha256.len(), 64);

    assert_eq!(
        fs::read_to_string(&output).expect("output is readable"),
        "id,language,text\n1,eng,\"Hello, world\"\n4,fra,Bonjour le monde\n"
    );
    assert_eq!(
        read_csv(&output),
        vec![
            vec![
                "1".to_string(),
                "eng".to_string(),
                "Hello, world".to_string()
            ],
            vec![
                "4".to_string(),
                "fra".to_string(),
                "Bonjour le monde".to_string()
            ],
        ]
    );
}

#[test]
fn preserves_text_verbatim() {
    let fixture = Fixture::new();
    let input = fixture.write_file(
        "csv/tatoeba.tsv",
        "1\teng\tHe said \"hi\" — and left\n2\tdeu\tGrüße aus Köln\n",
    );
    let output = fixture.csv_dir().join("tatoeba.csv");

    convert_tatoeba(&TatoebaOptions {
        input,
        output: output.clone(),
    })
    .expect("conversion succeeds");

    assert_eq!(
        read_csv(&output),
        vec![
            vec![
                "1".to_string(),
                "eng".to_string(),
                "He said \"hi\" — and left".to_string()
            ],
            vec![
                "2".to_string(),
                "deu".to_string(),
                "Grüße aus Köln".to_string()
            ],
        ]
    );
}

#[test]
fn rejects_rows_with_missing_fields() {
    let fixture = Fixture::new();
    let input = fixture.write_file("csv/tatoeba.tsv", "1\teng\n");

    let error = convert_tatoeba(&TatoebaOptions {
        input,
        output: fixture.csv_dir().join("tatoeba.csv"),
    })
    .expect_err("short rows are rejected");

    match error {
        Error::InvalidData(message) => assert!(message.contains("expected at least 3")),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rejects_a_dataset_without_valid_rows() {
    let fixture = Fixture::new();
    let input = fixture.write_file("csv/tatoeba.tsv", "1\t\\N\tignored\n");

    let error = convert_tatoeba(&TatoebaOptions {
        input,
        output: fixture.csv_dir().join("tatoeba.csv"),
    })
    .expect_err("an empty dataset is rejected");

    match error {
        Error::InvalidData(message) => assert!(message.contains("no valid rows")),
        other => panic!("unexpected error: {other}"),
    }
}

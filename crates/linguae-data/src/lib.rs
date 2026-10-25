//! Data preparation for the Linguae language detector.
//!
//! The crate converts the raw WiLI-2018 split files and the Tatoeba TSV into
//! the CSV layout consumed by training, and extracts a per-language word
//! dataset from WiLI for short-string detection.

pub mod error;
pub mod io;
pub mod paths;
pub mod stats;
pub mod tatoeba;
pub mod wili;
pub mod words;

pub use error::{Error, Result};
pub use io::{file_sha256, FileSummary};
pub use paths::{expand_tilde, find_data_root, Layout};
pub use stats::{LengthHistogram, LengthStats};
pub use tatoeba::{convert as convert_tatoeba, TatoebaOptions, TatoebaReport};
pub use wili::{convert as convert_wili, WiliOptions, WiliReport};
pub use words::{extract as extract_words, tokenize, WordOptions, WordReport, DEFAULT_MAX_WORDS};

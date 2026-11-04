//! Command-line entry point for the Linguae data pipeline.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use linguae_data::{
    convert_tatoeba, convert_wili, extract_words, Layout, TatoebaOptions, WiliOptions, WordOptions,
    DEFAULT_MAX_WORDS,
};

#[derive(Debug, Parser)]
#[command(
    name = "linguae-data",
    version,
    about = "Prepare the WiLI-2018, Tatoeba and word datasets used by Linguae."
)]
struct Cli {
    /// Path to the Linguae data directory (default: crates/linguae-data/data).
    #[arg(long, global = true, env = "LINGUAE_DATA_DIR", value_name = "PATH")]
    data_dir: Option<PathBuf>,

    /// Log level: error, warn, info, debug or trace.
    #[arg(
        long,
        global = true,
        env = "LINGUAE_LOG_LEVEL",
        default_value = "info",
        value_name = "LEVEL"
    )]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Convert the raw WiLI-2018 split files into data/csv/wili-2018.csv.
    Wili {
        /// Directory containing x_train.txt, y_train.txt, x_test.txt, y_test.txt.
        #[arg(long, env = "LINGUAE_WILI_DIR", value_name = "DIR")]
        input_dir: Option<PathBuf>,

        /// Destination CSV (default: data/csv/wili-2018.csv).
        #[arg(long, env = "LINGUAE_WILI_CSV", value_name = "PATH")]
        output: Option<PathBuf>,
    },

    /// Convert the Tatoeba TSV into data/csv/tatoeba.csv.
    Tatoeba {
        /// Headerless TSV with id, language and text columns.
        #[arg(long, env = "LINGUAE_TATOEBA_TSV", value_name = "PATH")]
        input: Option<PathBuf>,

        /// Destination CSV (default: data/csv/tatoeba.csv).
        #[arg(long, env = "LINGUAE_TATOEBA_CSV", value_name = "PATH")]
        output: Option<PathBuf>,
    },

    /// Extract the most common words per language from WiLI-2018.
    Words {
        /// Directory containing x_train.txt, y_train.txt, x_test.txt, y_test.txt.
        #[arg(long, env = "LINGUAE_WILI_DIR", value_name = "DIR")]
        input_dir: Option<PathBuf>,

        /// Destination CSV (default: data/csv/wili-2018-words.csv).
        #[arg(long, env = "LINGUAE_WILI_WORDS_CSV", value_name = "PATH")]
        output: Option<PathBuf>,

        /// Maximum words per language; 0 keeps every word.
        #[arg(
            long,
            env = "LINGUAE_MAX_WORDS_PER_LANGUAGE",
            default_value_t = DEFAULT_MAX_WORDS,
            value_name = "INT"
        )]
        max_words: usize,

        /// Use only the training split (avoids reusing test vocabulary).
        #[arg(long)]
        train_only: bool,
    },

    /// Run the WiLI, Tatoeba and word conversions in order.
    All {
        /// Maximum words per language; 0 keeps every word.
        #[arg(
            long,
            env = "LINGUAE_MAX_WORDS_PER_LANGUAGE",
            default_value_t = DEFAULT_MAX_WORDS,
            value_name = "INT"
        )]
        max_words: usize,

        /// Use only the training split for word extraction.
        #[arg(long)]
        train_only: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    env_logger::Builder::new()
        .parse_filters(&cli.log_level.to_ascii_lowercase())
        .init();

    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> linguae_data::Result<()> {
    let layout = Layout::discover(cli.data_dir.as_deref())?;

    match &cli.command {
        Command::Wili { input_dir, output } => {
            let options = WiliOptions {
                input_dir: input_dir.clone().unwrap_or_else(|| layout.wili_dir.clone()),
                output: output.clone().unwrap_or_else(|| layout.wili_csv.clone()),
            };
            print_wili(&options, &convert_wili(&options)?);
        }
        Command::Tatoeba { input, output } => {
            let options = TatoebaOptions {
                input: input.clone().unwrap_or_else(|| layout.tatoeba_tsv.clone()),
                output: output.clone().unwrap_or_else(|| layout.tatoeba_csv.clone()),
            };
            print_tatoeba(&options, &convert_tatoeba(&options)?);
        }
        Command::Words {
            input_dir,
            output,
            max_words,
            train_only,
        } => {
            let options = WordOptions {
                input_dir: input_dir.clone().unwrap_or_else(|| layout.wili_dir.clone()),
                output: output.clone().unwrap_or_else(|| layout.words_csv.clone()),
                max_words: *max_words,
                include_test: !train_only,
            };
            print_words(&options, &extract_words(&options)?);
        }
        Command::All {
            max_words,
            train_only,
        } => {
            let wili = WiliOptions {
                input_dir: layout.wili_dir.clone(),
                output: layout.wili_csv.clone(),
            };
            print_wili(&wili, &convert_wili(&wili)?);

            let tatoeba = TatoebaOptions {
                input: layout.tatoeba_tsv.clone(),
                output: layout.tatoeba_csv.clone(),
            };
            print_tatoeba(&tatoeba, &convert_tatoeba(&tatoeba)?);

            let words = WordOptions {
                input_dir: layout.wili_dir.clone(),
                output: layout.words_csv.clone(),
                max_words: *max_words,
                include_test: !train_only,
            };
            print_words(&words, &extract_words(&words)?);
        }
    }

    Ok(())
}

fn print_wili(options: &WiliOptions, report: &linguae_data::WiliReport) {
    println!(
        "WiLI-2018: {} rows ({} train + {} test), {} languages -> {} ({:.1} MiB)",
        report.rows,
        report.train_rows,
        report.test_rows,
        report.languages,
        options.output.display(),
        report.file.bytes as f64 / (1024.0 * 1024.0)
    );
}

fn print_tatoeba(options: &TatoebaOptions, report: &linguae_data::TatoebaReport) {
    println!(
        "Tatoeba: {} rows ({} dropped), {} languages -> {} ({:.1} MiB)",
        report.output_rows,
        report.dropped_rows,
        report.languages,
        options.output.display(),
        report.file.bytes as f64 / (1024.0 * 1024.0)
    );
}

fn print_words(options: &WordOptions, report: &linguae_data::WordReport) {
    println!(
        "Words: {} words across {} languages (max {}/language) -> {} ({:.1} MiB)",
        report.words,
        report.languages,
        report.max_words_per_language,
        options.output.display(),
        report.file.bytes as f64 / (1024.0 * 1024.0)
    );
}

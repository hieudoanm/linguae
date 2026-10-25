use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Size and content hash of a written file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSummary {
    pub bytes: u64,
    pub sha256: String,
}

/// Read a UTF-8 file into lines, preserving every line's content.
pub fn read_lines(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(line.map_err(|source| Error::Read {
            path: path.to_path_buf(),
            source,
        })?);
    }
    Ok(lines)
}

/// Write a CSV atomically: a temporary file is completed and synced before it
/// replaces the destination.
pub fn write_csv_atomic<F>(path: &Path, header: &[&str], write: F) -> Result<FileSummary>
where
    F: FnOnce(&mut csv::Writer<BufWriter<File>>) -> Result<()>,
{
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| Error::Write {
        path: parent.to_path_buf(),
        source,
    })?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output.csv");
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));

    let file = File::create(&temporary).map_err(|source| Error::Write {
        path: temporary.clone(),
        source,
    })?;
    let mut writer = csv::WriterBuilder::new().from_writer(BufWriter::new(file));
    writer.write_record(header).map_err(|source| Error::Csv {
        path: temporary.clone(),
        source,
    })?;
    write(&mut writer)?;
    writer.flush().map_err(|source| Error::Write {
        path: temporary.clone(),
        source,
    })?;
    writer
        .get_ref()
        .get_ref()
        .sync_all()
        .map_err(|source| Error::Write {
            path: temporary.clone(),
            source,
        })?;
    drop(writer);

    fs::rename(&temporary, path).map_err(|source| Error::Write {
        path: path.to_path_buf(),
        source,
    })?;

    let bytes = fs::metadata(path)
        .map_err(|source| Error::Read {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let sha256 = file_sha256(path)?;
    Ok(FileSummary { bytes, sha256 })
}

/// Compute a SHA-256 digest without loading the file into memory.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| Error::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| Error::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    let digest = hasher.finalize();
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(output)
}

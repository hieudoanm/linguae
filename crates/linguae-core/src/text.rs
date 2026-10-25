use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;

/// Python's `\s` (and `str.isspace`) also treats the file/group/record/unit
/// separators as whitespace, while Rust's `char::is_whitespace` does not.
fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

/// NFC-normalise and collapse whitespace without changing case or accents.
pub fn normalize_text(text: &str) -> String {
    let normalized: String = text.nfc().collect();
    let mut output = String::with_capacity(normalized.len());
    let mut pending_space = false;

    for character in normalized.chars() {
        if is_python_whitespace(character) {
            if !output.is_empty() {
                pending_space = true;
            }
        } else {
            if pending_space {
                output.push(' ');
                pending_space = false;
            }
            output.push(character);
        }
    }

    output
}

/// Encode one string as character IDs, mapping unknown characters to `UNK_ID`.
pub fn encode_text(text: &str, vocabulary: &HashMap<char, u32>, max_length: usize) -> Vec<u32> {
    let normalized = normalize_text(text);
    let mut encoded: Vec<u32> = normalized
        .chars()
        .take(max_length)
        .map(|character| vocabulary.get(&character).copied().unwrap_or(UNK_ID))
        .collect();

    if encoded.is_empty() {
        encoded.push(UNK_ID);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary() -> HashMap<char, u32> {
        [('a', 2u32), ('b', 3), ('é', 4)].into_iter().collect()
    }

    #[test]
    fn normalizes_nfc_and_collapses_whitespace() {
        assert_eq!(normalize_text("  a\t\tb\n e\u{301}  "), "a b é");
    }

    #[test]
    fn treats_python_control_whitespace_as_whitespace() {
        assert_eq!(normalize_text("a\u{1c}\u{1d}b"), "a b");
    }

    #[test]
    fn keeps_internal_punctuation_and_case() {
        assert_eq!(normalize_text("Hello,  World!"), "Hello, World!");
    }

    #[test]
    fn encodes_unknown_characters_as_unk() {
        assert_eq!(encode_text("abz", &vocabulary(), 256), vec![2, 3, UNK_ID]);
    }

    #[test]
    fn encodes_empty_text_as_a_single_unk() {
        assert_eq!(encode_text("   ", &vocabulary(), 256), vec![UNK_ID]);
    }

    #[test]
    fn truncates_by_unicode_code_point() {
        assert_eq!(encode_text("aéé", &vocabulary(), 2), vec![2, 4]);
    }
}

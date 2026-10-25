use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::source::Record;

pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const SPECIAL_TOKENS: [&str; 2] = [PAD_TOKEN, UNK_TOKEN];

/// Character vocabulary with reserved pad/unknown ids.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token -> id, including the special tokens. Used for JSON export.
    pub token_to_id: BTreeMap<String, u32>,
    /// Single-character -> id. Used for encoding.
    pub char_to_id: HashMap<char, u32>,
}

impl Vocabulary {
    /// Build the vocabulary from the training split only.
    pub fn from_texts<'a, I>(texts: I) -> Self
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut characters = BTreeSet::new();
        for text in texts {
            characters.extend(text.chars());
        }

        let mut token_to_id = BTreeMap::new();
        for (index, token) in SPECIAL_TOKENS.iter().enumerate() {
            token_to_id.insert((*token).to_string(), index as u32);
        }

        let mut char_to_id = HashMap::with_capacity(characters.len());
        for (index, character) in (SPECIAL_TOKENS.len() as u32..).zip(characters) {
            token_to_id.insert(character.to_string(), index);
            char_to_id.insert(character, index);
        }

        Self {
            token_to_id,
            char_to_id,
        }
    }

    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    pub fn encode(&self, text: &str, max_length: usize) -> Vec<u32> {
        linguae_core::text::encode_text(text, &self.char_to_id, max_length)
    }

    /// Share of characters in `records` that are not in the vocabulary.
    pub fn unknown_character_rate(&self, records: &[Record]) -> f64 {
        let mut total = 0usize;
        let mut unknown = 0usize;
        for record in records {
            for character in record.text.chars() {
                total += 1;
                if !self.char_to_id.contains_key(&character) {
                    unknown += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            unknown as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_special_ids_and_sorts_characters() {
        let vocabulary = Vocabulary::from_texts(["cab", "ba"]);
        assert_eq!(vocabulary.token_to_id[PAD_TOKEN], PAD_ID);
        assert_eq!(vocabulary.token_to_id[UNK_TOKEN], UNK_ID);
        assert_eq!(vocabulary.char_to_id[&'a'], 2);
        assert_eq!(vocabulary.char_to_id[&'b'], 3);
        assert_eq!(vocabulary.char_to_id[&'c'], 4);
        assert_eq!(vocabulary.len(), 5);
    }

    #[test]
    fn encodes_unknown_characters_as_unk() {
        let vocabulary = Vocabulary::from_texts(["a"]);
        assert_eq!(vocabulary.encode("az", 256), vec![2, UNK_ID]);
    }
}

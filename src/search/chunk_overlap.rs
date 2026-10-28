//! Phrase search across the chunks of a split message.
//!
//! FAD splits a message over 4 MiB into chunk rows, and each row is one
//! lexical document. A phrase that crosses the cut between two chunks would
//! then match neither. So the document for chunk k also indexes the head of
//! chunk k+1 after its own text (a forward overlap). Every document then
//! starts with its own text, so the content-prefix field and the stored
//! preview, which frankensearch derives from the start of the document, stay
//! the chunk's own; and the final chunk, often short, gets no overlap.
//!
//! Limit: a phrase of at most [`MAX_CROSS_CHUNK_PHRASE_BYTES`] (UTF-8 bytes,
//! separators included) that crosses one chunk cut matches. A longer one does
//! not. Every chunk but the last is a full chunk (4 MiB), so a chunk that has
//! a successor is always longer than the overlap.
//!
//! The head ends right before a separator: a char that no token of the cass
//! analyzer can contain (not alphanumeric and not `-`). Chunk k plus its head
//! is one contiguous slice of the message, so the tokens of the document from
//! the first separator in chunk k on are exactly the tokens the whole message
//! has over the same bytes. Where the head has no separator near the limit
//! (one very long token, such as base64), it ends at the last separator
//! before the limit, or is empty. That loses no indexed token: a run with no
//! separator for 4 KiB is longer than the analyzer's longest token, so it is
//! never indexed, and any token after it lies out of phrase range.
//!
//! Chunk k is searchable without its head until chunk k+1 is stored and the
//! incremental publish refreshes chunk k's document.

use franken_agent_detection::MESSAGE_CHUNK_KEY;
use serde_json::Value;

/// The longest phrase that still matches when it crosses a chunk cut.
pub const MAX_CROSS_CHUNK_PHRASE_BYTES: usize = 64 * 1024;

/// Longest token the cass analyzer keeps (frankensearch-quill
/// `CASS_MAX_TOKEN_BYTES`); the head covers this much beyond the phrase limit
/// so a phrase that ends in such a token still matches.
const MAX_TOKEN_BYTES: usize = 256;

/// The least the head covers when a separator lies near the limit.
const HEAD_LIMIT_BYTES: usize = MAX_CROSS_CHUNK_PHRASE_BYTES + MAX_TOKEN_BYTES;

/// How far past the limit the head may run, to end before a separator.
const SEPARATOR_SEARCH_BYTES: usize = 4 * 1024;

/// The longest head [`overlap_head`] returns.
pub const MAX_OVERLAP_BYTES: usize = HEAD_LIMIT_BYTES + SEPARATOR_SEARCH_BYTES;

/// Bump when the overlap rule changes, so lexical indexes built with the old
/// rule are rebuilt. 2: forward head instead of backward tail.
pub const CHUNK_OVERLAP_VERSION: u32 = 2;

/// Whether no cass token can contain `ch` (the analyzer splits on it).
fn is_separator(ch: char) -> bool {
    let alphanumeric = if ch.is_ascii() {
        ch.is_ascii_alphanumeric()
    } else {
        ch.is_alphanumeric()
    };
    !alphanumeric && ch != '-'
}

/// The part of `next` (the text of chunk k+1) that the document for chunk k
/// indexes after its own text: at least the first [`HEAD_LIMIT_BYTES`] bytes
/// when a separator lies within [`SEPARATOR_SEARCH_BYTES`] after them, ending
/// right before a separator. A `next` no longer than the limit is returned
/// whole: it ends the message.
#[must_use]
pub fn overlap_head(next: &str) -> &str {
    if next.len() <= HEAD_LIMIT_BYTES {
        return next;
    }
    // Round the limit up, so a separator char (an emoji, say) that straddles
    // it cannot end the head short of the limit.
    let limit = ceil_char_boundary(next, HEAD_LIMIT_BYTES);
    let latest = floor_char_boundary(next, HEAD_LIMIT_BYTES + SEPARATOR_SEARCH_BYTES);
    // The first separator at or after the limit.
    if let Some((at, _)) = next[limit..latest]
        .char_indices()
        .find(|(_, ch)| is_separator(*ch))
    {
        return &next[..limit + at];
    }
    // Else the last separator before it: coverage is then below the limit.
    match next[..limit]
        .char_indices()
        .rev()
        .find(|(_, ch)| is_separator(*ch))
    {
        Some((at, _)) => &next[..at],
        None => "",
    }
}

fn floor_char_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn ceil_char_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

/// The text a lexical document for a message indexes: the message itself,
/// then the head of the chunk after it when there is one.
#[must_use]
pub fn lexical_text(content: &str, next_head: Option<&str>) -> String {
    match next_head {
        Some(head) if !head.is_empty() => {
            let mut text = String::with_capacity(content.len() + head.len());
            text.push_str(content);
            text.push_str(head);
            text
        }
        _ => content.to_string(),
    }
}

/// Whether `next` is the chunk right after `cur` in the same split message:
/// same group and total size, the next ordinal, and byte ranges that meet.
#[must_use]
pub fn chunk_follows(cur_extra: &Value, next_extra: &Value) -> bool {
    let (Some(cur), Some(next)) = (
        cur_extra.get(MESSAGE_CHUNK_KEY),
        next_extra.get(MESSAGE_CHUNK_KEY),
    ) else {
        return false;
    };
    let field = |record: &Value, name: &str| record.get(name).cloned();
    let number = |record: &Value, name: &str| record.get(name).and_then(Value::as_u64);
    field(cur, "group").is_some()
        && field(cur, "group") == field(next, "group")
        && number(cur, "total_bytes").is_some()
        && number(cur, "total_bytes") == number(next, "total_bytes")
        && number(cur, "ordinal").map(|ordinal| ordinal + 1) == number(next, "ordinal")
        && number(cur, "byte_end").is_some()
        && number(cur, "byte_end") == number(next, "byte_start")
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch::quill::schema::Analyzer;
    use frankensearch::quill::scribe::{CASS_MAX_TOKEN_BYTES, CassAnalyzer, TokenAnalyzer};

    /// Tokens of `text` as the lexical index's content analyzer emits them:
    /// (text, byte offset, position).
    fn tokens(text: &str) -> Vec<(String, usize, u32)> {
        let mut out = Vec::new();
        CassAnalyzer::default().analyze(Analyzer::CassHyphenNormalize, text, &mut |token| {
            out.push((token.text.clone(), token.offset_from, token.position));
        });
        out
    }

    fn record(ordinal: u64, start: u64, end: u64) -> Value {
        serde_json::json!({MESSAGE_CHUNK_KEY: {
            "group": "g", "ordinal": ordinal, "byte_start": start,
            "byte_end": end, "total_bytes": 100
        }})
    }

    /// Tokens of `chunk + head` from the first separator in `chunk` on equal
    /// the whole message's tokens over the same bytes, with positions shifted
    /// by the same amount.
    fn assert_equivalent(message: &str, cut: usize) {
        let (chunk, next) = message.split_at(cut);
        let head = overlap_head(next);
        let doc = lexical_text(chunk, Some(head));
        let start = chunk.find(is_separator).expect("a separator in the chunk");
        let end = cut + head.len();
        let from = |tokens: Vec<(String, usize, u32)>, base: usize| {
            let kept: Vec<_> = tokens
                .into_iter()
                .filter(|(_, at, _)| *at > base + start && *at < base + end)
                .collect();
            let first = kept.first().map_or(0, |token| token.2);
            kept.into_iter()
                .map(|(text, at, position)| (text, at - base, position - first))
                .collect::<Vec<_>>()
        };
        let whole = from(tokens(message), 0);
        let indexed = from(tokens(&doc), 0);
        assert!(!indexed.is_empty());
        assert_eq!(indexed, whole);
        // Nothing after the head can join its last token.
        assert!(next[head.len()..].starts_with(is_separator) || head.len() == next.len());
    }

    #[test]
    fn the_head_covers_the_limit_and_ends_before_a_separator() {
        let next = "word ".repeat(30_000);
        let head = overlap_head(&next);
        assert!(head.len() >= HEAD_LIMIT_BYTES, "{}", head.len());
        assert!(head.len() <= MAX_OVERLAP_BYTES);
        assert!(next[head.len()..].starts_with(' '));
    }

    #[test]
    fn chunk_and_head_tokens_equal_the_whole_message_tokens() {
        // Hyphens, CJK runs without spaces, multibyte chars, punctuation as
        // the only separator, and a cut inside a word.
        let units = [
            "alpha-beta 日本語のテキスト café 😀 x-ray-42 ",
            "日本語のテキストが続く。次の文も続く、",
            "path/to/file.rs:42,next_token;",
        ];
        for unit in units {
            let message = unit.repeat(6_000);
            for cut in [70_001, 70_002, 70_003, 70_017] {
                let cut = floor_char_boundary(&message, cut);
                assert_equivalent(&message, cut);
            }
        }
    }

    #[test]
    fn a_hyphen_or_cjk_run_never_ends_the_head() {
        // At the limit there is only a hyphenated word, then a CJK run; the
        // head must run on to the space after them.
        let prefix = "a ".repeat(HEAD_LIMIT_BYTES / 2 - 2);
        let next = format!("{prefix}x-ray-yankee-zulu日本語日本語 tail");
        let head = overlap_head(&next);
        assert!(
            head.ends_with("日本語日本語"),
            "{}",
            &head[head.len() - 40..]
        );
        assert!(next[head.len()..].starts_with(' '));
    }

    #[test]
    fn the_head_limit_includes_the_longest_token() {
        assert_eq!(MAX_TOKEN_BYTES, CASS_MAX_TOKEN_BYTES);
    }

    #[test]
    fn a_multibyte_char_at_the_limit_is_never_split() {
        for pad in 0..4 {
            let next = format!("{}{}", "a".repeat(pad), "😀 ".repeat(40_000));
            let head = overlap_head(&next);
            assert!(next.starts_with(head));
            assert!(head.len() >= HEAD_LIMIT_BYTES);
        }
    }

    #[test]
    fn a_head_without_a_separator_is_empty_and_a_short_chunk_is_whole() {
        let next = "A".repeat(MAX_OVERLAP_BYTES * 2);
        assert_eq!(overlap_head(&next), "");
        assert_eq!(lexical_text("own", Some("")), "own");
        assert_eq!(overlap_head("short last chunk"), "short last chunk");
        assert_eq!(lexical_text("own ", Some("next")), "own next");
    }

    #[test]
    fn only_the_next_chunk_of_the_same_message_follows() {
        assert!(chunk_follows(&record(0, 0, 40), &record(1, 40, 80)));
        assert!(
            !chunk_follows(&record(0, 0, 40), &record(2, 40, 80)),
            "ordinal gap"
        );
        assert!(
            !chunk_follows(&record(0, 0, 40), &record(1, 41, 80)),
            "byte gap"
        );
        let mut other = record(1, 40, 80);
        other[MESSAGE_CHUNK_KEY]["group"] = Value::from("h");
        assert!(!chunk_follows(&record(0, 0, 40), &other), "other group");
        let mut total = record(1, 40, 80);
        total[MESSAGE_CHUNK_KEY]["total_bytes"] = Value::from(99);
        assert!(!chunk_follows(&record(0, 0, 40), &total), "other total");
        assert!(!chunk_follows(&record(0, 0, 40), &serde_json::json!({})));
    }
}

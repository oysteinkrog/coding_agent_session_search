//! Phrase search across the chunks of a split message.
//!
//! FAD splits a message over 4 MiB into chunk rows, and each row is one
//! lexical document. A phrase that crosses the cut between two chunks would
//! then match neither. So the document for chunk k+1 also indexes the tail of
//! chunk k, at most [`MAX_CROSS_CHUNK_PHRASE_BYTES`] of it (a backward
//! overlap). Backward, because a long message can go out in parts: chunk k+1
//! may not exist yet when chunk k is indexed, but chunk k always exists when
//! chunk k+1 is.
//!
//! Limit: a phrase of at most [`MAX_CROSS_CHUNK_PHRASE_BYTES`] (UTF-8 bytes,
//! whitespace included) that crosses one chunk cut matches. A longer one does
//! not. Every chunk but the last is a full chunk (4 MiB), so the chunk before
//! any other chunk is always longer than the overlap.
//!
//! The overlap starts right after a whitespace character. The cass tokenizer
//! treats whitespace as a separator, so the tokens of the overlap are exactly
//! the tokens the whole message has from that point on. Where the tail has
//! no whitespace near the limit (one very long token, such as base64), the
//! overlap starts at the first whitespace after it, or is empty.

use franken_agent_detection::MESSAGE_CHUNK_KEY;
use serde_json::Value;

/// The longest phrase that still matches when it crosses a chunk cut.
pub const MAX_CROSS_CHUNK_PHRASE_BYTES: usize = 64 * 1024;

/// How far before the limit the overlap may start, to begin at whitespace.
const RESTART_SEARCH_BYTES: usize = 4 * 1024;

/// Bump when the overlap rule changes, so lexical indexes built with the old
/// rule are rebuilt.
pub const CHUNK_OVERLAP_VERSION: u32 = 1;

/// The part of `prev` (the text of chunk k) that the document for chunk k+1
/// indexes before its own text: at least the last
/// [`MAX_CROSS_CHUNK_PHRASE_BYTES`] bytes when a whitespace character lies
/// within [`RESTART_SEARCH_BYTES`] before them, starting right after
/// whitespace. A `prev` no longer than the limit is returned whole.
#[must_use]
pub fn overlap_tail(prev: &str) -> &str {
    if prev.len() <= MAX_CROSS_CHUNK_PHRASE_BYTES {
        // Only a last chunk is this short, and no chunk follows a last one.
        // Take it whole rather than guess.
        return prev;
    }
    let limit = prev.len() - MAX_CROSS_CHUNK_PHRASE_BYTES;
    let earliest = limit.saturating_sub(RESTART_SEARCH_BYTES);
    // The last whitespace at or before the limit, searched backwards.
    if let Some(start) = whitespace_end_in(prev, earliest, limit, true) {
        return &prev[start..];
    }
    // Else the first whitespace after it: coverage is then below the limit.
    match whitespace_end_in(prev, limit, prev.len(), false) {
        Some(start) => &prev[start..],
        None => "",
    }
}

/// The byte just after a whitespace char whose start lies in `from..to`
/// (char boundaries are found inside), the last one when `last` is set.
fn whitespace_end_in(text: &str, from: usize, to: usize, last: bool) -> Option<usize> {
    let from = floor_char_boundary(text, from);
    let to = floor_char_boundary(text, to);
    let window = &text[from..to];
    let mut found = window
        .char_indices()
        .filter(|(_, ch)| ch.is_whitespace())
        .map(|(at, ch)| from + at + ch.len_utf8());
    if last { found.last() } else { found.next() }
}

fn floor_char_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// The text a lexical document for a message indexes: the message itself,
/// after the overlap from the chunk before it when there is one.
#[must_use]
pub fn lexical_text(prev_tail: Option<&str>, content: &str) -> String {
    match prev_tail {
        Some(tail) if !tail.is_empty() => {
            let mut text = String::with_capacity(tail.len() + content.len());
            text.push_str(tail);
            text.push_str(content);
            text
        }
        _ => content.to_string(),
    }
}

/// Whether `prev` is the chunk right before `cur` in the same split message:
/// same group and total size, the next ordinal, and byte ranges that meet.
#[must_use]
pub fn chunk_follows(prev_extra: &Value, cur_extra: &Value) -> bool {
    let (Some(prev), Some(cur)) = (
        prev_extra.get(MESSAGE_CHUNK_KEY),
        cur_extra.get(MESSAGE_CHUNK_KEY),
    ) else {
        return false;
    };
    let field = |record: &Value, name: &str| record.get(name).cloned();
    let number = |record: &Value, name: &str| record.get(name).and_then(Value::as_u64);
    field(prev, "group").is_some()
        && field(prev, "group") == field(cur, "group")
        && number(prev, "total_bytes").is_some()
        && number(prev, "total_bytes") == number(cur, "total_bytes")
        && number(prev, "ordinal").map(|ordinal| ordinal + 1) == number(cur, "ordinal")
        && number(prev, "byte_end").is_some()
        && number(prev, "byte_end") == number(cur, "byte_start")
}

/// Whether `extra` marks a chunk that continues an earlier one, so its
/// document needs the overlap from the chunk before.
#[must_use]
pub fn continues_chunk(extra: &Value) -> bool {
    extra
        .get(MESSAGE_CHUNK_KEY)
        .and_then(|record| record.get("ordinal"))
        .and_then(Value::as_u64)
        .is_some_and(|ordinal| ordinal > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch::quill::schema::Analyzer;
    use frankensearch::quill::scribe::{CassAnalyzer, TokenAnalyzer};

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

    #[test]
    fn the_overlap_covers_the_limit_and_starts_after_whitespace() {
        let prev = "word ".repeat(30_000);
        let tail = overlap_tail(&prev);
        assert!(tail.len() >= MAX_CROSS_CHUNK_PHRASE_BYTES, "{}", tail.len());
        assert!(tail.len() <= MAX_CROSS_CHUNK_PHRASE_BYTES + RESTART_SEARCH_BYTES);
        let start = prev.len() - tail.len();
        assert!(prev[..start].ends_with(' '));
    }

    #[test]
    fn overlap_tokens_equal_the_whole_message_tokens_from_the_restart() {
        // Hyphens, CJK runs, multibyte chars, and a cut inside a word.
        let unit = "alpha-beta 日本語のテキスト café 😀 x-ray-42 ";
        let prev = unit.repeat(5_000);
        let cur = "ndgame final-word 文字列 end";
        let tail = overlap_tail(&prev);
        assert!(!tail.is_empty());
        let start = prev.len() - tail.len();
        let whole = format!("{prev}{cur}");
        let from_restart: Vec<_> = tokens(&whole)
            .into_iter()
            .filter(|(_, at, _)| *at >= start)
            .collect();
        let first_position = from_restart[0].2;
        let expected: Vec<_> = from_restart
            .into_iter()
            .map(|(text, at, position)| (text, at - start, position - first_position))
            .collect();
        assert_eq!(tokens(&lexical_text(Some(tail), cur)), expected);
    }

    #[test]
    fn a_multibyte_char_at_the_limit_is_never_split() {
        for pad in 0..4 {
            let prev = format!("{}{}", "a".repeat(pad), "😀 ".repeat(40_000));
            let tail = overlap_tail(&prev);
            assert!(prev.ends_with(tail));
            assert!(tail.len() >= MAX_CROSS_CHUNK_PHRASE_BYTES);
        }
    }

    #[test]
    fn a_tail_without_whitespace_has_no_overlap() {
        let prev = "A".repeat(MAX_CROSS_CHUNK_PHRASE_BYTES * 2);
        assert_eq!(overlap_tail(&prev), "");
        assert_eq!(lexical_text(Some(""), "next"), "next");
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
        assert!(!chunk_follows(&serde_json::json!({}), &record(1, 40, 80)));
        assert!(continues_chunk(&record(1, 40, 80)));
        assert!(!continues_chunk(&record(0, 0, 40)));
    }
}

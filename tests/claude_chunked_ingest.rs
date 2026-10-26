//! Local patches (localbuilds recipe "cass"): a large Claude Code session is
//! stored in parts, and a grown session is ingested from its tail. Both must
//! leave the archive, search and display exactly as one full parse would.

use assert_cmd::Command;
use coding_agent_search::franken_sync::compat::RowExt;
use coding_agent_search::storage::sqlite::SqliteStorage;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::TempDir;

const SESSION_REL: &str = ".claude/projects/-work-proj/11111111-2222-3333-4444-555555555555.jsonl";

fn line(i: usize) -> String {
    let role = if i % 2 == 0 { "user" } else { "assistant" };
    // Every message is searchable on its own number; the first and last carry
    // extra needles.
    let mut text = format!("chunkprobe message {i} word{i}");
    if i == 0 {
        text.push_str(" firstneedle");
    }
    if i == 59 {
        text.push_str(" lastneedle");
    }
    format!(
        r#"{{"type":"{role}","cwd":"/work/proj","sessionId":"11111111-2222-3333-4444-555555555555","gitBranch":"main","timestamp":"2025-12-01T10:{:02}:{:02}Z","message":{{"role":"{role}","content":"{text}"}}}}"#,
        (i / 60) % 60,
        i % 60
    )
}

fn session_lines(count: usize, title_after: Option<usize>) -> Vec<String> {
    let mut lines = Vec::new();
    for i in 0..count {
        lines.push(line(i));
        if Some(i) == title_after {
            lines.push(r#"{"type":"ai-title","title":"Chunk probe session"}"#.to_string());
        }
    }
    lines
}

fn write_lines(path: &Path, lines: &[String]) {
    let mut text = lines.join("\n");
    text.push('\n');
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

fn cmd(home: &Path, data_dir: &Path, chunk_messages: Option<usize>) -> Command {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("cass"));
    cmd.env("CODING_AGENT_SEARCH_NO_UPDATE_PROMPT", "1")
        .env("HOME", home)
        .env("XDG_DATA_HOME", home.join(".local/share"))
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("CODEX_HOME", home.join(".codex"))
        .env("CASS_AUTO_REFRESH", "0")
        .env("CASS_IGNORE_SOURCES_CONFIG", "1")
        .env("CASS_DATA_DIR", data_dir)
        .timeout(Duration::from_secs(180));
    match chunk_messages {
        Some(n) => cmd.env("FAD_CLAUDE_CHUNK_MAX_MESSAGES", n.to_string()),
        None => cmd.env_remove("FAD_CLAUDE_CHUNK_MAX_MESSAGES"),
    };
    cmd
}

fn run_ok(mut cmd: Command, args: &[&str]) -> Value {
    let output = cmd.args(args).output().unwrap();
    assert!(
        output.status.success(),
        "cass {args:?} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or(Value::Null)
}

fn index(home: &Path, data_dir: &Path, chunk_messages: Option<usize>, full: bool) -> Value {
    let mut args = vec!["index", "--json"];
    if full {
        args.push("--full");
    }
    run_ok(cmd(home, data_dir, chunk_messages), &args)
}

/// Conversation and message rows as stored, without row ids.
fn archive(data_dir: &Path) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    let storage = SqliteStorage::open_readonly(&data_dir.join("agent_search.db")).unwrap();
    let conversations = storage
        .raw()
        .query(
            "SELECT external_id, title, source_path, CAST(started_at AS TEXT), CAST(ended_at AS TEXT), \
             CAST(workspace_id AS TEXT) FROM conversations ORDER BY external_id",
        )
        .unwrap()
        .iter()
        .map(|row| (0..6).map(|i| row.get_typed::<Option<String>>(i).unwrap().unwrap_or_default()).collect())
        .collect();
    let messages = storage
        .raw()
        .query(
            "SELECT CAST(m.idx AS TEXT), m.role, m.content, CAST(m.created_at AS TEXT) \
             FROM messages m JOIN conversations c ON c.id = m.conversation_id ORDER BY c.external_id, m.idx",
        )
        .unwrap()
        .iter()
        .map(|row| (0..4).map(|i| row.get_typed::<Option<String>>(i).unwrap().unwrap_or_default()).collect())
        .collect();
    storage.close().unwrap();
    (conversations, messages)
}

/// Search hits with run-specific fields (scores, timings, data dir) removed.
fn hits(home: &Path, data_dir: &Path, query: &str) -> Vec<Value> {
    let payload = run_ok(
        cmd(home, data_dir, None),
        &["search", query, "--mode", "lexical", "--no-maintenance", "--json", "--limit", "100"],
    );
    let mut hits: Vec<Value> = payload["hits"].as_array().cloned().unwrap_or_default();
    for hit in &mut hits {
        if let Some(object) = hit.as_object_mut() {
            object.retain(|key, _| {
                !key.contains("score") && !key.contains("rank") && !key.ends_with("_ms")
            });
        }
    }
    hits.sort_by_key(|hit| hit.to_string());
    hits
}

fn sessions(home: &Path, data_dir: &Path) -> Value {
    let mut payload = run_ok(cmd(home, data_dir, None), &["sessions", "--json", "--limit", "10"]);
    strip_keys(&mut payload, &["data_dir", "db_path", "elapsed_ms", "generated_at"]);
    payload
}

fn strip_keys(value: &mut Value, keys: &[&str]) {
    match value {
        Value::Object(object) => {
            object.retain(|key, _| !keys.contains(&key.as_str()));
            object.values_mut().for_each(|v| strip_keys(v, keys));
        }
        Value::Array(items) => items.iter_mut().for_each(|v| strip_keys(v, keys)),
        _ => {}
    }
}

struct Fixture {
    _temp: TempDir,
    home: PathBuf,
    session: PathBuf,
}

fn fixture(lines: &[String]) -> Fixture {
    let temp = TempDir::new().unwrap();
    let home = temp.path().join("home");
    let session = home.join(SESSION_REL);
    write_lines(&session, lines);
    Fixture {
        _temp: temp,
        home,
        session,
    }
}

#[test]
fn a_session_stored_in_parts_searches_and_displays_like_one_stored_whole() {
    // 60 messages in parts of 7; Claude names the session near the end.
    let f = fixture(&session_lines(60, Some(55)));
    let whole = f.home.parent().unwrap().join("data-whole");
    let parts = f.home.parent().unwrap().join("data-parts");
    index(&f.home, &whole, Some(100_000), true);
    index(&f.home, &parts, Some(7), true);

    let (conv_whole, msgs_whole) = archive(&whole);
    let (conv_parts, msgs_parts) = archive(&parts);
    assert_eq!(conv_whole.len(), 1, "one conversation: {conv_whole:?}");
    assert_eq!(conv_parts, conv_whole, "one row with the same title, times and workspace");
    assert_eq!(msgs_whole.len(), 60);
    assert_eq!(msgs_parts, msgs_whole, "same messages and idx");
    assert_eq!(conv_parts[0][1], "Chunk probe session");

    for query in ["firstneedle", "lastneedle", "word30", "chunkprobe"] {
        assert_eq!(
            hits(&f.home, &parts, query),
            hits(&f.home, &whole, query),
            "search for {query}"
        );
    }
    assert!(!hits(&f.home, &parts, "lastneedle").is_empty());
    assert_eq!(sessions(&f.home, &parts), sessions(&f.home, &whole));
}

#[test]
fn a_grown_session_is_ingested_from_its_tail_and_matches_a_fresh_full_index() {
    let all = session_lines(60, Some(50));
    let f = fixture(&all[..40]);
    let tail = f.home.parent().unwrap().join("data-tail");
    index(&f.home, &tail, Some(7), false);
    assert_eq!(archive(&tail).1.len(), 40);

    // Append 21 lines (20 messages and the ai-title) and index again.
    write_lines(&f.session, &all);
    index(&f.home, &tail, Some(7), false);

    let fresh = f.home.parent().unwrap().join("data-fresh");
    index(&f.home, &fresh, None, true);
    assert_eq!(archive(&tail), archive(&fresh), "rows equal a fresh full index");
    assert_eq!(hits(&f.home, &tail, "lastneedle"), hits(&f.home, &fresh, "lastneedle"));

    // Nothing changed: a third run parses nothing and changes nothing.
    index(&f.home, &tail, Some(7), false);
    assert_eq!(archive(&tail), archive(&fresh));
}

#[test]
fn a_rewritten_session_falls_back_to_a_full_parse() {
    let all = session_lines(30, None);
    let f = fixture(&all[..20]);
    let data = f.home.parent().unwrap().join("data-rewrite");
    index(&f.home, &data, None, false);

    // Rewrite the bytes just before the saved offset (same length), then
    // append. The prefix hash no longer matches, so the file is parsed again
    // from the start and the archive keeps one copy of each message.
    let mut rewritten = all.clone();
    rewritten[19] = rewritten[19].replace("chunkprobe", "chunkPROBE");
    write_lines(&f.session, &rewritten);
    index(&f.home, &data, None, false);
    let (_, messages) = archive(&data);
    assert_eq!(messages.len(), 30, "no duplicates after the fallback");
    let idx: Vec<_> = messages.iter().map(|m| m[0].clone()).collect();
    let expected: Vec<_> = (0..30).map(|i| i.to_string()).collect();
    assert_eq!(idx, expected);

    // A truncated file (shorter than the saved offset) also parses in full
    // without error.
    write_lines(&f.session, &all[..10]);
    index(&f.home, &data, None, false);
}

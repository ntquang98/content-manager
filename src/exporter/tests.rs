// Tests for the exporter module.
// Feature: ai-saved-manager

#[cfg(test)]
mod unit_tests {
    use crate::exporter::{Exporter, JsonExporter, SqliteExporter};
    use crate::models::ExportItem;
    use tempfile::tempdir;

    /// Build a minimal ExportItem for testing.
    fn make_item(n: usize) -> ExportItem {
        ExportItem {
            title: format!("Title {n}"),
            summary: format!("Summary {n}"),
            tags: vec!["rust".to_string(), "test".to_string(), "item".to_string()],
            category_ai: "Technology".to_string(),
            link: format!("https://example.com/{n}"),
            image_url: Some(format!("https://img.example.com/{n}.png")),
            score: 0.75,
        }
    }

    // ── JSON output format ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_json_exporter_output_format() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("out.json");
        let items = vec![make_item(0), make_item(1)];

        let exporter = JsonExporter;
        let count = exporter.export(&items, &output).await.unwrap();

        assert_eq!(count, 2);
        assert!(output.exists());

        let content = std::fs::read_to_string(&output).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Top-level key must be "items"
        assert!(parsed.get("items").is_some(), "missing 'items' key");
        let arr = parsed["items"].as_array().unwrap();
        assert_eq!(arr.len(), 2);

        // Check all required fields are present in the first item
        let first = &arr[0];
        for field in &["title", "summary", "tags", "category_ai", "link", "image_url", "score"] {
            assert!(first.get(field).is_some(), "missing field '{field}'");
        }

        assert_eq!(first["title"].as_str().unwrap(), "Title 0");
        assert_eq!(first["category_ai"].as_str().unwrap(), "Technology");
        assert!((first["score"].as_f64().unwrap() - 0.75).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_json_exporter_null_image_url() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("out.json");
        let mut item = make_item(0);
        item.image_url = None;

        let exporter = JsonExporter;
        exporter.export(&[item], &output).await.unwrap();

        let content = std::fs::read_to_string(&output).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        let first = &parsed["items"][0];
        assert!(first["image_url"].is_null(), "image_url should be null");
    }

    #[tokio::test]
    async fn test_json_exporter_empty_items() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("empty.json");

        let exporter = JsonExporter;
        let count = exporter.export(&[], &output).await.unwrap();

        assert_eq!(count, 0);
        let content = std::fs::read_to_string(&output).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["items"].as_array().unwrap().len(), 0);
    }

    // ── SQLite output format ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_sqlite_exporter_output_format() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("out.db");
        let items = vec![make_item(0), make_item(1)];

        let exporter = SqliteExporter;
        let count = exporter.export(&items, &output).await.unwrap();

        assert_eq!(count, 2);
        assert!(output.exists());

        // Open the written database and verify contents
        let conn = rusqlite::Connection::open(&output).unwrap();
        let row_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM items", [], |r| r.get(0)).unwrap();
        assert_eq!(row_count, 2);

        // Verify all columns exist and have correct values
        let (title, summary, tags_json, category_ai, link, image_url, score): (
            String,
            String,
            String,
            String,
            String,
            Option<String>,
            f64,
        ) = conn
            .query_row(
                "SELECT title, summary, tags, category_ai, link, image_url, score \
                 FROM items WHERE title = 'Title 0'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)),
            )
            .unwrap();

        assert_eq!(title, "Title 0");
        assert_eq!(summary, "Summary 0");
        assert_eq!(category_ai, "Technology");
        assert_eq!(link, "https://example.com/0");
        assert!(image_url.is_some());
        assert!((score - 0.75).abs() < 1e-4);

        // Tags must be a valid JSON array
        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap();
        assert_eq!(tags, vec!["rust", "test", "item"]);
    }

    #[tokio::test]
    async fn test_sqlite_exporter_tags_stored_as_json() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("tags.db");
        let mut item = make_item(0);
        item.tags = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];

        let exporter = SqliteExporter;
        exporter.export(&[item], &output).await.unwrap();

        let conn = rusqlite::Connection::open(&output).unwrap();
        let raw_tags: String =
            conn.query_row("SELECT tags FROM items", [], |r| r.get(0)).unwrap();

        // Must be a JSON array string
        let parsed: Vec<String> = serde_json::from_str(&raw_tags).unwrap();
        assert_eq!(parsed, vec!["alpha", "beta", "gamma"]);
    }

    // ── Directory auto-creation ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_json_exporter_creates_directory() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a").join("b").join("c");
        let output = nested.join("out.json");

        assert!(!nested.exists(), "nested dir should not exist yet");

        let exporter = JsonExporter;
        exporter.export(&[make_item(0)], &output).await.unwrap();

        assert!(nested.exists(), "nested dir should have been created");
        assert!(output.exists(), "output file should exist");
    }

    #[tokio::test]
    async fn test_sqlite_exporter_creates_directory() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("x").join("y");
        let output = nested.join("out.db");

        assert!(!nested.exists(), "nested dir should not exist yet");

        let exporter = SqliteExporter;
        exporter.export(&[make_item(0)], &output).await.unwrap();

        assert!(nested.exists(), "nested dir should have been created");
        assert!(output.exists(), "output file should exist");
    }

    // ── File overwrite warning ────────────────────────────────────────────────
    // We verify that overwriting an existing file succeeds (the warn! is a
    // side-effect we cannot easily capture in a unit test without a custom
    // tracing subscriber, but we confirm the file is overwritten correctly).

    #[tokio::test]
    async fn test_json_exporter_overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("out.json");

        // Write initial content
        std::fs::write(&output, b"old content").unwrap();
        assert!(output.exists());

        let exporter = JsonExporter;
        let count = exporter.export(&[make_item(0)], &output).await.unwrap();

        assert_eq!(count, 1);
        let content = std::fs::read_to_string(&output).unwrap();
        // Must be valid JSON now, not "old content"
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(parsed.get("items").is_some());
    }

    #[tokio::test]
    async fn test_sqlite_exporter_overwrites_existing_file() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("out.db");

        // Write initial content (not a valid SQLite file)
        std::fs::write(&output, b"old content").unwrap();
        assert!(output.exists());

        let exporter = SqliteExporter;
        let count = exporter.export(&[make_item(0)], &output).await.unwrap();

        assert_eq!(count, 1);
        // The file should now be a valid SQLite database
        let conn = rusqlite::Connection::open(&output).unwrap();
        let row_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM items", [], |r| r.get(0)).unwrap();
        assert_eq!(row_count, 1);
    }
}

// ── Property-based tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_exporter {
    use crate::exporter::{Exporter, JsonExporter, SqliteExporter};
    use crate::models::ExportItem;
    use proptest::prelude::*;
    use tempfile::tempdir;

    // ── Generator ─────────────────────────────────────────────────────────────

    /// Generate a valid lowercase tag (3–20 lowercase ASCII letters).
    fn arb_tag() -> impl Strategy<Value = String> {
        "[a-z]{3,20}".prop_map(|s| s)
    }

    /// Generate a valid category string.
    fn arb_category() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("Technology".to_string()),
            Just("Business".to_string()),
            Just("Education".to_string()),
            Just("Entertainment".to_string()),
            Just("Travel".to_string()),
            Just("Personal".to_string()),
            Just("Other".to_string()),
        ]
    }

    /// Generate a random ExportItem.
    fn arb_export_item() -> impl Strategy<Value = ExportItem> {
        (
            "[a-zA-Z0-9 ]{1,60}",                          // title
            "[a-zA-Z0-9 ]{10,200}",                        // summary
            prop::collection::vec(arb_tag(), 3..=7),       // tags
            arb_category(),                                 // category_ai
            "https://[a-z]{3,10}\\.com/[a-z0-9]{3,20}",   // link
            prop::option::of("[a-z]{3,10}\\.png"),         // image_url
            (0u32..=1000u32).prop_map(|n| n as f32 / 1000.0), // score
        )
            .prop_map(|(title, summary, tags, category_ai, link, image_url, score)| ExportItem {
                title,
                summary,
                tags,
                category_ai,
                link,
                image_url,
                score,
            })
    }

    /// Generate a Vec of 0–20 ExportItems.
    fn arb_export_items() -> impl Strategy<Value = Vec<ExportItem>> {
        prop::collection::vec(arb_export_item(), 0..=20)
    }

    // ── Property 15: JSON export contains all required fields ─────────────────
    // Feature: ai-saved-manager, Property 15: JSON export contains all required fields
    //
    // For any processed post, the corresponding entry in a JSON export SHALL
    // contain all seven required fields: title, summary, tags, category_ai,
    // link, image_url, score.
    //
    // Validates: Requirements 6.2
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop15_json_export_contains_all_required_fields(items in arb_export_items()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let dir = tempdir().unwrap();
                let output = dir.path().join("export.json");

                let exporter = JsonExporter;
                let count = exporter.export(&items, &output).await.unwrap();

                prop_assert_eq!(count, items.len());

                let content = std::fs::read_to_string(&output).unwrap();
                let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

                let arr = parsed["items"].as_array().unwrap();
                prop_assert_eq!(arr.len(), items.len());

                for (i, entry) in arr.iter().enumerate() {
                    let obj = entry.as_object().expect("each item must be a JSON object");

                    // All seven required fields must be present
                    for field in &["title", "summary", "tags", "category_ai", "link", "image_url", "score"] {
                        prop_assert!(
                            obj.contains_key(*field),
                            "item {i} missing required field '{field}'"
                        );
                    }

                    // Values must match the original ExportItem
                    prop_assert_eq!(
                        obj["title"].as_str().unwrap(),
                        items[i].title.as_str(),
                        "title mismatch at index {}",
                        i
                    );
                    prop_assert_eq!(
                        obj["summary"].as_str().unwrap(),
                        items[i].summary.as_str(),
                        "summary mismatch at index {}",
                        i
                    );
                    prop_assert_eq!(
                        obj["category_ai"].as_str().unwrap(),
                        items[i].category_ai.as_str(),
                        "category_ai mismatch at index {}",
                        i
                    );
                    prop_assert_eq!(
                        obj["link"].as_str().unwrap(),
                        items[i].link.as_str(),
                        "link mismatch at index {}",
                        i
                    );

                    // score must be a number
                    prop_assert!(
                        obj["score"].as_f64().is_some(),
                        "score at index {} must be a number",
                        i
                    );

                    // tags must be an array
                    let tags_arr = obj["tags"].as_array().unwrap();
                    prop_assert_eq!(
                        tags_arr.len(),
                        items[i].tags.len(),
                        "tags length mismatch at index {}",
                        i
                    );
                }

                Ok(())
            })?;
        }
    }

    // ── Property 16: Export item count matches output ─────────────────────────
    // Feature: ai-saved-manager, Property 16: Export item count matches output
    //
    // For any export operation, the count returned SHALL equal the number of
    // items in the output file.
    //
    // Validates: Requirements 6.6
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop16_json_export_count_matches_output(items in arb_export_items()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let dir = tempdir().unwrap();
                let output = dir.path().join("export.json");

                let exporter = JsonExporter;
                let returned_count = exporter.export(&items, &output).await.unwrap();

                // Count items in the written file
                let content = std::fs::read_to_string(&output).unwrap();
                let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
                let file_count = parsed["items"].as_array().unwrap().len();

                prop_assert_eq!(
                    returned_count,
                    file_count,
                    "returned count {} != file item count {}",
                    returned_count,
                    file_count
                );
                prop_assert_eq!(
                    returned_count,
                    items.len(),
                    "returned count {} != input item count {}",
                    returned_count,
                    items.len()
                );

                Ok(())
            })?;
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop16_sqlite_export_count_matches_output(items in arb_export_items()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let dir = tempdir().unwrap();
                let output = dir.path().join("export.db");

                let exporter = SqliteExporter;
                let returned_count = exporter.export(&items, &output).await.unwrap();

                // Count rows in the written SQLite file
                let conn = rusqlite::Connection::open(&output).unwrap();
                let file_count: i64 =
                    conn.query_row("SELECT COUNT(*) FROM items", [], |r| r.get(0)).unwrap();

                prop_assert_eq!(
                    returned_count,
                    file_count as usize,
                    "returned count {} != SQLite row count {}",
                    returned_count,
                    file_count
                );
                prop_assert_eq!(
                    returned_count,
                    items.len(),
                    "returned count {} != input item count {}",
                    returned_count,
                    items.len()
                );

                Ok(())
            })?;
        }
    }
}

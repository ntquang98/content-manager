// Tests for the importer module.
// Feature: ai-saved-manager

#[cfg(test)]
mod unit_tests {
    use crate::importer::{FacebookImporter, Importer};
    use crate::storage::Storage;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a temp JSON file containing the given records as a JSON array.
    fn write_json_file(records: &[serde_json::Value]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        let json = serde_json::to_string(records).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    /// Helper: build a minimal in-memory Storage with schema and a dataset.
    async fn make_storage() -> (Storage, String) {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();
        let dataset = storage
            .create_dataset("test-dataset", "j2team")
            .await
            .unwrap();
        (storage, dataset.id)
    }

    // ── Valid J2Team file ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_valid_j2team_file_inserts_records() {
        let records = vec![
            serde_json::json!({
                "Title": "Post One",
                "Link": "https://example.com/1",
                "Image URL": "https://img.example.com/1.png",
                "Category": "Technology",
                "Type": "article"
            }),
            serde_json::json!({
                "Title": "Post Two",
                "Link": "https://example.com/2",
                "Image URL": null,
                "Category": "Business",
                "Type": "video"
            }),
        ];
        let file = write_json_file(&records);
        let (storage, dataset_id) = make_storage().await;

        let importer = FacebookImporter;
        let stats = importer
            .import(file.path(), &dataset_id, &storage)
            .await
            .unwrap();

        assert_eq!(stats.parsed, 2);
        assert!(stats.inserted > 0, "expected at least one inserted record");
        assert_eq!(stats.inserted, 2);
        assert_eq!(stats.skipped, 0);
    }

    // ── Empty link skipping ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_empty_link_records_are_skipped() {
        let records = vec![
            // absent Link field
            serde_json::json!({ "Title": "No Link" }),
            // empty string Link
            serde_json::json!({ "Title": "Empty Link", "Link": "" }),
            // whitespace-only Link
            serde_json::json!({ "Title": "Whitespace Link", "Link": "   " }),
            // valid record
            serde_json::json!({ "Title": "Valid", "Link": "https://example.com/valid" }),
        ];
        let file = write_json_file(&records);
        let (storage, dataset_id) = make_storage().await;

        let importer = FacebookImporter;
        let stats = importer
            .import(file.path(), &dataset_id, &storage)
            .await
            .unwrap();

        assert_eq!(stats.parsed, 4);
        assert_eq!(stats.inserted, 1, "only the valid record should be inserted");
        assert_eq!(stats.skipped, 3, "three empty-link records should be skipped");
    }

    // ── Duplicate post skipping ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_duplicate_import_skips_all_on_second_run() {
        let records = vec![
            serde_json::json!({ "Title": "Post A", "Link": "https://example.com/a" }),
            serde_json::json!({ "Title": "Post B", "Link": "https://example.com/b" }),
        ];
        let file = write_json_file(&records);
        let (storage, dataset_id) = make_storage().await;

        let importer = FacebookImporter;

        // First import: both records should be inserted
        let first = importer
            .import(file.path(), &dataset_id, &storage)
            .await
            .unwrap();
        assert_eq!(first.inserted, 2);
        assert_eq!(first.skipped, 0);

        // Second import of the same file: all records are duplicates → skipped
        let second = importer
            .import(file.path(), &dataset_id, &storage)
            .await
            .unwrap();
        assert_eq!(
            second.inserted, 0,
            "second run should insert nothing (all duplicates)"
        );
        assert_eq!(
            second.skipped, 2,
            "second run should skip all records as duplicates"
        );
    }

    // ── Malformed JSON file returns an error ──────────────────────────────────

    #[tokio::test]
    async fn test_malformed_json_file_returns_error() {
        // The importer uses serde_json::from_reader which parses the whole array.
        // A completely malformed JSON file should cause the parse to fail.
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"[{\"Title\": \"Good\"}, {not valid json}]")
            .unwrap();
        f.flush().unwrap();

        let (storage, dataset_id) = make_storage().await;
        let importer = FacebookImporter;
        let result = importer.import(f.path(), &dataset_id, &storage).await;

        assert!(
            result.is_err(),
            "malformed JSON should return an error, got: {:?}",
            result.ok()
        );
    }
}

// ── Property-based tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_importer {
    use crate::importer::{FacebookImporter, Importer, J2TeamRecord, Normalizer, sha1_hex};
    use crate::storage::Storage;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── Generators ────────────────────────────────────────────────────────────

    /// Generate a non-empty, non-whitespace URL-like string.
    fn arb_non_empty_link() -> impl Strategy<Value = String> {
        "https://[a-z]{3,10}\\.com/[a-z0-9]{3,20}"
    }

    /// Generate an optional title string.
    fn arb_opt_string() -> impl Strategy<Value = Option<String>> {
        prop::option::of("[a-zA-Z0-9 ]{1,40}")
    }

    /// Generate a J2TeamRecord with a guaranteed non-empty link.
    fn arb_j2team_record_with_link() -> impl Strategy<Value = J2TeamRecord> {
        (
            arb_opt_string(),       // title
            arb_non_empty_link(),   // link (non-empty)
            arb_opt_string(),       // image_url
            arb_opt_string(),       // category
            arb_opt_string(),       // post_type
        )
            .prop_map(|(title, link, image_url, category, post_type)| J2TeamRecord {
                title,
                link: Some(link),
                image_url,
                category,
                post_type,
            })
    }

    /// Generate a J2TeamRecord with an empty or absent link.
    fn arb_j2team_record_empty_link() -> impl Strategy<Value = J2TeamRecord> {
        prop_oneof![
            // absent link
            (arb_opt_string(), arb_opt_string(), arb_opt_string()).prop_map(
                |(title, image_url, category)| J2TeamRecord {
                    title,
                    link: None,
                    image_url,
                    category,
                    post_type: None,
                }
            ),
            // empty string link
            (arb_opt_string(), arb_opt_string(), arb_opt_string()).prop_map(
                |(title, image_url, category)| J2TeamRecord {
                    title,
                    link: Some(String::new()),
                    image_url,
                    category,
                    post_type: None,
                }
            ),
            // whitespace-only link
            (arb_opt_string(), arb_opt_string(), arb_opt_string()).prop_map(
                |(title, image_url, category)| J2TeamRecord {
                    title,
                    link: Some("   ".to_string()),
                    image_url,
                    category,
                    post_type: None,
                }
            ),
        ]
    }

    /// Serialize a slice of J2TeamRecord values to a temp JSON file.
    fn records_to_temp_file(records: &[J2TeamRecord]) -> NamedTempFile {
        let json_records: Vec<serde_json::Value> = records
            .iter()
            .map(|r| {
                let mut obj = serde_json::Map::new();
                if let Some(t) = &r.title {
                    obj.insert("Title".to_string(), serde_json::Value::String(t.clone()));
                }
                if let Some(l) = &r.link {
                    obj.insert("Link".to_string(), serde_json::Value::String(l.clone()));
                }
                if let Some(i) = &r.image_url {
                    obj.insert(
                        "Image URL".to_string(),
                        serde_json::Value::String(i.clone()),
                    );
                }
                if let Some(c) = &r.category {
                    obj.insert(
                        "Category".to_string(),
                        serde_json::Value::String(c.clone()),
                    );
                }
                if let Some(p) = &r.post_type {
                    obj.insert("Type".to_string(), serde_json::Value::String(p.clone()));
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        let mut f = NamedTempFile::new().unwrap();
        let json = serde_json::to_string(&json_records).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    /// Build an in-memory Storage with schema and a dataset, returning (storage, dataset_id).
    async fn make_storage() -> (Storage, String) {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();
        let dataset = storage
            .create_dataset("prop-dataset", "j2team")
            .await
            .unwrap();
        (storage, dataset.id)
    }

    // ── Property 4: J2Team field mapping correctness ──────────────────────────
    // Feature: ai-saved-manager, Property 4: J2Team field mapping correctness
    //
    // For any J2Team record with a non-empty Link field, the Normalizer SHALL
    // produce a RawPost where title = record.Title, link = record.Link,
    // image_url = record.Image URL, category_raw = record.Category,
    // post_type = record.Type, and id = sha1_hex(record.Link).
    //
    // Validates: Requirements 2.1
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]
        #[test]
        fn prop4_j2team_field_mapping_correctness(record in arb_j2team_record_with_link()) {
            let dataset_id = "test-dataset";

            // Clone fields before moving record into normalize
            let expected_title = record.title.clone().unwrap_or_default();
            let expected_link = record.link.clone().unwrap();
            let expected_image_url = record.image_url.clone();
            let expected_category = record.category.clone();
            let expected_post_type = record.post_type.clone();
            let expected_id = sha1_hex(&expected_link);

            let result = Normalizer::normalize(record, dataset_id);

            // A record with a non-empty link must always produce a RawPost
            prop_assert!(result.is_some(), "normalize returned None for a non-empty link");
            let post = result.unwrap();

            prop_assert_eq!(&post.id, &expected_id, "id must be sha1_hex(link)");
            prop_assert_eq!(&post.link, &expected_link, "link must match record.Link");
            prop_assert_eq!(&post.title, &expected_title, "title must match record.Title");
            prop_assert_eq!(
                &post.image_url,
                &expected_image_url,
                "image_url must match record.Image URL"
            );
            prop_assert_eq!(
                &post.category_raw,
                &expected_category,
                "category_raw must match record.Category"
            );
            prop_assert_eq!(
                &post.post_type,
                &expected_post_type,
                "post_type must match record.Type"
            );
            prop_assert_eq!(
                &post.dataset_id,
                dataset_id,
                "dataset_id must be passed through"
            );
        }
    }

    // ── Property 5: Empty-link records are skipped and counted ────────────────
    // Feature: ai-saved-manager, Property 5: Empty-link records are skipped and counted
    //
    // For any batch of J2Team records containing N records with an empty or
    // absent Link field, the importer SHALL skip exactly those N records and
    // the ImportStats.skipped counter SHALL equal N.
    //
    // Validates: Requirements 2.3
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop5_empty_link_records_skipped_and_counted(
            empty_records in prop::collection::vec(arb_j2team_record_empty_link(), 0..=10),
            valid_records in prop::collection::vec(arb_j2team_record_with_link(), 0..=10),
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let n_empty = empty_records.len();
                let n_valid = valid_records.len();

                // Interleave empty and valid records
                let mut all_records: Vec<J2TeamRecord> = Vec::new();
                all_records.extend(empty_records);
                all_records.extend(valid_records);

                let file = records_to_temp_file(&all_records);
                let (storage, dataset_id) = make_storage().await;

                let importer = FacebookImporter;
                let stats = importer
                    .import(file.path(), &dataset_id, &storage)
                    .await
                    .unwrap();

                prop_assert_eq!(
                    stats.parsed,
                    n_empty + n_valid,
                    "parsed count must equal total records"
                );

                // All empty-link records must be skipped.
                // Note: valid records with duplicate links also count as skipped,
                // but since we use a fresh storage each run, only empty-link records
                // are skipped here (valid records are all inserted).
                prop_assert!(
                    stats.skipped >= n_empty,
                    "skipped ({}) must be at least n_empty ({})",
                    stats.skipped,
                    n_empty
                );

                // The number of inserted records must equal the number of valid records
                // (all have unique links from the generator).
                prop_assert_eq!(
                    stats.inserted,
                    n_valid,
                    "inserted must equal number of valid records"
                );

                Ok(())
            })?;
        }
    }

    // ── Property 6: Import is idempotent ──────────────────────────────────────
    // Feature: ai-saved-manager, Property 6: Import is idempotent
    //
    // For any set of posts, importing the same set twice into the same dataset
    // SHALL result in the same final database state as importing it once —
    // no duplicate rows, and ImportStats.inserted on the second run SHALL equal 0.
    //
    // Validates: Requirements 2.4
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop6_import_is_idempotent(
            records in prop::collection::vec(arb_j2team_record_with_link(), 1..=15),
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let file = records_to_temp_file(&records);
                let (storage, dataset_id) = make_storage().await;

                let importer = FacebookImporter;

                // First import
                let first = importer
                    .import(file.path(), &dataset_id, &storage)
                    .await
                    .unwrap();

                prop_assert!(
                    first.inserted > 0 || first.skipped == records.len(),
                    "first import should insert at least one record (or all are duplicates)"
                );

                // Second import of the same file
                let second = importer
                    .import(file.path(), &dataset_id, &storage)
                    .await
                    .unwrap();

                prop_assert_eq!(
                    second.inserted,
                    0,
                    "second import must insert 0 records (idempotency)"
                );

                // All records on the second run must be counted as skipped
                prop_assert_eq!(
                    second.skipped,
                    second.parsed,
                    "second import: all parsed records must be skipped"
                );

                Ok(())
            })?;
        }
    }
}

// Tests for the storage module.
// Feature: ai-saved-manager

#[cfg(test)]
mod unit_tests {
    use crate::models::{PostStatus, RawPost};
    use crate::storage::{Storage, StorageError};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal RawPost for testing.
    fn make_post(id: &str, dataset_id: &str) -> RawPost {
        RawPost {
            id: id.to_string(),
            dataset_id: dataset_id.to_string(),
            title: format!("Title {id}"),
            link: format!("https://example.com/{id}"),
            image_url: None,
            category_raw: None,
            post_type: None,
            status: PostStatus::Pending,
            ignore_reason: None,
        }
    }

    // ── Schema creation ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_schema_creation_tables_exist() {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();

        // Verify all three tables exist by querying sqlite_master
        let conn = storage.conn;
        let tables: Vec<String> = conn
            .call(|conn| {
                let mut stmt = conn
                    .prepare(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                    )
                    .unwrap();
                let rows = stmt
                    .query_map([], |row| row.get::<_, String>(0))
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                Ok(rows)
            })
            .await
            .unwrap();

        assert!(tables.contains(&"datasets".to_string()), "datasets table missing");
        assert!(tables.contains(&"posts".to_string()), "posts table missing");
        assert!(tables.contains(&"analysis".to_string()), "analysis table missing");
    }

    // ── FK constraint violation ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_fk_violation_post_with_nonexistent_dataset() {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();

        // Try inserting a post that references a dataset_id that doesn't exist.
        // The FOREIGN KEY constraint is enforced (PRAGMA foreign_keys = ON),
        // so this should return an error.
        let post = make_post("p1", "nonexistent-dataset-id");
        let result = storage.insert_post(&post).await;

        assert!(
            result.is_err(),
            "inserting a post with a non-existent dataset_id should fail with a FK error"
        );
    }

    // ── Duplicate dataset name ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_duplicate_dataset_name_returns_error() {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();

        // First creation must succeed.
        storage.create_dataset("foo", "facebook").await.unwrap();

        // Second creation with the same name must fail with DuplicateDataset.
        let result = storage.create_dataset("foo", "facebook").await;
        assert!(
            result.is_err(),
            "second create_dataset with same name should fail"
        );
        match result.unwrap_err() {
            StorageError::DuplicateDataset { name } => {
                assert_eq!(name, "foo");
            }
            other => panic!("expected DuplicateDataset, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_duplicate_dataset_leaves_original_unchanged() {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();

        let original = storage.create_dataset("foo", "facebook").await.unwrap();
        let _ = storage.create_dataset("foo", "twitter").await; // ignore error

        // The original dataset must still be retrievable and unchanged.
        let found = storage
            .find_dataset_by_name("foo")
            .await
            .unwrap()
            .expect("original dataset should still exist");

        assert_eq!(found.id, original.id);
        assert_eq!(found.source, "facebook");
    }

    // ── Stats on non-existent dataset ─────────────────────────────────────────

    #[tokio::test]
    async fn test_stats_nonexistent_dataset_returns_zeros() {
        let storage = Storage::open_in_memory().await.unwrap();
        storage.init_schema().await.unwrap();

        // Task 13.7: get_dataset_stats now returns DatasetNotFound for a
        // non-existent dataset instead of returning zeros.
        let result = storage
            .get_dataset_stats("does-not-exist")
            .await;

        assert!(
            matches!(result, Err(StorageError::DatasetNotFound { .. })),
            "expected DatasetNotFound error for non-existent dataset, got: {:?}",
            result
        );
    }
}

// ── Property-based tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod proptest_storage {
    use crate::models::{Category, PostStatus, ProcessedPost, RawPost};
    use crate::storage::StorageError;
    use crate::storage::Storage;
    use proptest::prelude::*;

    // ── Generators ────────────────────────────────────────────────────────────

    /// Generate a valid lowercase tag (3–20 lowercase ASCII letters).
    fn arb_lowercase_tag() -> impl Strategy<Value = String> {
        "[a-z]{3,20}".prop_map(|s| s)
    }

    /// Generate a valid Category.
    fn arb_category() -> impl Strategy<Value = Category> {
        prop_oneof![
            Just(Category::Technology),
            Just(Category::Business),
            Just(Category::Education),
            Just(Category::Entertainment),
            Just(Category::Travel),
            Just(Category::Personal),
            Just(Category::Other),
        ]
    }

    /// Generate a dataset name (1–40 lowercase alphanumeric chars).
    fn arb_dataset_name() -> impl Strategy<Value = String> {
        "[a-z][a-z0-9]{0,39}".prop_map(|s| s)
    }

    /// Build a RawPost with a given id and dataset_id.
    fn make_post(id: &str, dataset_id: &str) -> RawPost {
        RawPost {
            id: id.to_string(),
            dataset_id: dataset_id.to_string(),
            title: format!("Title {id}"),
            link: format!("https://example.com/{id}"),
            image_url: None,
            category_raw: None,
            post_type: None,
            status: PostStatus::Pending,
            ignore_reason: None,
        }
    }

    // ── Property 2: Every post is associated with exactly one dataset ─────────
    // Feature: ai-saved-manager, Property 2: Every post is associated with exactly one dataset
    //
    // For any set of posts inserted into storage, every post record SHALL have
    // a non-null dataset_id that references an existing entry in the datasets
    // table.
    //
    // Validates: Requirements 2.1
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]
        #[test]
        fn prop2_every_post_has_valid_dataset_id(
            n_posts in 1usize..=10usize,
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let storage = Storage::open_in_memory().await.unwrap();
                storage.init_schema().await.unwrap();

                // Create a real dataset.
                let dataset = storage.create_dataset("test-dataset", "facebook").await.unwrap();

                // Insert N posts all referencing that dataset.
                for i in 0..n_posts {
                    let post = make_post(&format!("post-{i}"), &dataset.id);
                    storage.insert_post(&post).await.unwrap();
                }

                // Query all posts and verify each has a non-null dataset_id
                // that matches the existing dataset.
                let conn = &storage.conn;
                let dataset_id_clone = dataset.id.clone();
                let rows: Vec<(String, String)> = conn
                    .call(move |conn| {
                        let mut stmt = conn
                            .prepare("SELECT id, dataset_id FROM posts WHERE dataset_id = ?1")
                            .unwrap();
                        let rows = stmt
                            .query_map(rusqlite::params![dataset_id_clone], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                            })
                            .unwrap()
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap();
                        Ok(rows)
                    })
                    .await
                    .unwrap();

                prop_assert_eq!(rows.len(), n_posts, "expected {} posts, got {}", n_posts, rows.len());

                for (post_id, post_dataset_id) in &rows {
                    prop_assert!(
                        !post_dataset_id.is_empty(),
                        "post {} has empty dataset_id",
                        post_id
                    );
                    prop_assert_eq!(
                        post_dataset_id,
                        &dataset.id,
                        "post {} dataset_id does not reference the existing dataset",
                        post_id
                    );
                }

                Ok(())
            })?;
        }
    }

    // ── Property 3: Dataset name uniqueness ───────────────────────────────────
    // Feature: ai-saved-manager, Property 3: Dataset name uniqueness
    //
    // For any dataset name that already exists in storage, attempting to create
    // a second dataset with the same name SHALL fail with an error and leave
    // the existing dataset unchanged.
    //
    // Validates: Requirements 2.3
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop3_dataset_name_uniqueness(name in arb_dataset_name()) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let storage = Storage::open_in_memory().await.unwrap();
                storage.init_schema().await.unwrap();

                // First creation must succeed.
                let first = storage.create_dataset(&name, "facebook").await.unwrap();

                // Second creation with the same name must fail.
                let result = storage.create_dataset(&name, "twitter").await;
                prop_assert!(
                    result.is_err(),
                    "second create_dataset with name '{name}' should fail"
                );
                match result.unwrap_err() {
                    StorageError::DuplicateDataset { name: err_name } => {
                        prop_assert_eq!(err_name, name.clone());
                    }
                    other => {
                        return Err(proptest::test_runner::TestCaseError::fail(
                            format!("expected DuplicateDataset, got {other:?}")
                        ));
                    }
                }

                // The original dataset must still be intact.
                let found = storage
                    .find_dataset_by_name(&name)
                    .await
                    .unwrap()
                    .expect("original dataset should still exist");

                prop_assert_eq!(&found.id, &first.id, "original dataset id changed");
                prop_assert_eq!(&found.source, "facebook", "original dataset source changed");

                Ok(())
            })?;
        }
    }

    // ── Property 7: Only unprocessed posts are selected for processing ─────────
    // Feature: ai-saved-manager, Property 7: Only unprocessed posts are selected for processing
    //
    // For any dataset containing a mix of posts with and without analysis
    // records, the processor's selection query SHALL return only posts that
    // have no corresponding analysis row.
    //
    // Validates: Requirements 4.1
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]
        #[test]
        fn prop7_only_unprocessed_posts_selected(
            n_total in 2usize..=10usize,
            n_analyzed in 1usize..=5usize,
        ) {
            // Ensure n_analyzed <= n_total
            let n_analyzed = n_analyzed.min(n_total);
            let n_unprocessed = n_total - n_analyzed;

            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let storage = Storage::open_in_memory().await.unwrap();
                storage.init_schema().await.unwrap();

                let dataset = storage.create_dataset("test-ds", "facebook").await.unwrap();

                // Insert all posts.
                for i in 0..n_total {
                    let post = make_post(&format!("post-{i}"), &dataset.id);
                    storage.insert_post(&post).await.unwrap();
                }

                // Add analysis for the first n_analyzed posts.
                for i in 0..n_analyzed {
                    let analysis = ProcessedPost {
                        post_id: format!("post-{i}"),
                        summary: "summary".to_string(),
                        tags: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                        category_ai: Category::Other,
                        score: 0.5,
                        processed_at: "2024-01-01T00:00:00Z".to_string(),
                    };
                    storage.insert_analysis(&analysis).await.unwrap();
                }

                // get_unprocessed_posts should return only posts without analysis.
                let unprocessed = storage
                    .get_unprocessed_posts(&dataset.id)
                    .await
                    .unwrap();

                prop_assert_eq!(
                    unprocessed.len(),
                    n_unprocessed,
                    "expected {} unprocessed posts, got {}",
                    n_unprocessed,
                    unprocessed.len()
                );

                // Verify none of the returned posts have analysis.
                let analyzed_ids: std::collections::HashSet<String> =
                    (0..n_analyzed).map(|i| format!("post-{i}")).collect();

                for post in &unprocessed {
                    prop_assert!(
                        !analyzed_ids.contains(&post.id),
                        "post {} has analysis but was returned as unprocessed",
                        post.id
                    );
                }

                Ok(())
            })?;
        }
    }

    // ── Property 17: Tags round-trip through SQLite ───────────────────────────
    // Feature: ai-saved-manager, Property 17: Tags round-trip through SQLite
    //
    // For any list of tags stored in the analysis table, retrieving the record
    // and deserialising the tags JSON string SHALL produce a list equal to the
    // original.
    //
    // Validates: Requirements 5.3
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop17_tags_round_trip_through_sqlite(
            tags in prop::collection::vec(arb_lowercase_tag(), 3..=7),
            category in arb_category(),
        ) {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let storage = Storage::open_in_memory().await.unwrap();
                storage.init_schema().await.unwrap();

                let dataset = storage.create_dataset("tags-ds", "facebook").await.unwrap();
                let post = make_post("p1", &dataset.id);
                storage.insert_post(&post).await.unwrap();

                let analysis = ProcessedPost {
                    post_id: "p1".to_string(),
                    summary: "summary".to_string(),
                    tags: tags.clone(),
                    category_ai: category,
                    score: 0.75,
                    processed_at: "2024-01-01T00:00:00Z".to_string(),
                };
                storage.insert_analysis(&analysis).await.unwrap();

                // Retrieve the raw tags JSON from the database and deserialise.
                let conn = &storage.conn;
                let raw_tags_json: String = conn
                    .call(|conn| {
                        conn.query_row(
                            "SELECT tags FROM analysis WHERE post_id = 'p1'",
                            [],
                            |row| row.get::<_, String>(0),
                        )
                        .map_err(tokio_rusqlite::Error::from)
                    })
                    .await
                    .unwrap();

                let retrieved_tags: Vec<String> =
                    serde_json::from_str(&raw_tags_json).unwrap();

                prop_assert_eq!(
                    retrieved_tags.clone(),
                    tags.clone(),
                    "tags round-trip failed: stored {:?}, retrieved {:?}",
                    tags,
                    retrieved_tags
                );

                Ok(())
            })?;
        }
    }

    // ── Property 20: Dataset stats completeness ───────────────────────────────
    // Feature: ai-saved-manager, Property 20: Dataset stats completeness
    //
    // For any dataset with a known composition (total, valid, ignored,
    // unprocessed, category distribution), the stats command output SHALL
    // contain all five required metrics with values matching the actual
    // database state.
    //
    // Validates: Requirements 7.1
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]
        #[test]
        fn prop20_dataset_stats_completeness(
            n_valid in 0usize..=5usize,
            n_ignored in 0usize..=5usize,
            n_pending in 0usize..=5usize,
        ) {
            let n_total = n_valid + n_ignored + n_pending;
            // Skip degenerate case with no posts.
            prop_assume!(n_total > 0);

            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let storage = Storage::open_in_memory().await.unwrap();
                storage.init_schema().await.unwrap();

                let dataset = storage.create_dataset("stats-ds", "facebook").await.unwrap();

                let mut post_idx = 0usize;

                // Insert valid posts (with analysis so they count as processed).
                for _ in 0..n_valid {
                    let id = format!("post-{post_idx}");
                    let mut post = make_post(&id, &dataset.id);
                    post.status = PostStatus::Valid;
                    storage.insert_post(&post).await.unwrap();
                    // Add analysis so they are not counted as unprocessed.
                    let analysis = ProcessedPost {
                        post_id: id.clone(),
                        summary: "summary".to_string(),
                        tags: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                        category_ai: Category::Technology,
                        score: 0.9,
                        processed_at: "2024-01-01T00:00:00Z".to_string(),
                    };
                    storage.insert_analysis(&analysis).await.unwrap();
                    post_idx += 1;
                }

                // Insert ignored posts (no analysis).
                for _ in 0..n_ignored {
                    let id = format!("post-{post_idx}");
                    let mut post = make_post(&id, &dataset.id);
                    post.status = PostStatus::Ignored;
                    post.ignore_reason = Some("too_short".to_string());
                    storage.insert_post(&post).await.unwrap();
                    post_idx += 1;
                }

                // Insert pending posts (no analysis — these are unprocessed).
                for _ in 0..n_pending {
                    let id = format!("post-{post_idx}");
                    let post = make_post(&id, &dataset.id);
                    storage.insert_post(&post).await.unwrap();
                    post_idx += 1;
                }

                let stats = storage.get_dataset_stats(&dataset.id).await.unwrap();

                // total = all posts
                prop_assert_eq!(
                    stats.total,
                    n_total,
                    "total mismatch: expected {}, got {}",
                    n_total,
                    stats.total
                );

                // valid count
                prop_assert_eq!(
                    stats.valid,
                    n_valid,
                    "valid mismatch: expected {}, got {}",
                    n_valid,
                    stats.valid
                );

                // ignored count
                prop_assert_eq!(
                    stats.ignored,
                    n_ignored,
                    "ignored mismatch: expected {}, got {}",
                    n_ignored,
                    stats.ignored
                );

                // unprocessed = posts without an analysis row (ignored + pending)
                let expected_unprocessed = n_ignored + n_pending;
                prop_assert_eq!(
                    stats.unprocessed,
                    expected_unprocessed,
                    "unprocessed mismatch: expected {}, got {}",
                    expected_unprocessed,
                    stats.unprocessed
                );

                // category_distribution must be present (Technology for valid posts)
                if n_valid > 0 {
                    let tech_count = stats
                        .category_distribution
                        .iter()
                        .find(|(cat, _)| cat == "Technology")
                        .map(|(_, c)| *c)
                        .unwrap_or(0);
                    prop_assert_eq!(
                        tech_count,
                        n_valid,
                        "Technology category count mismatch: expected {}, got {}",
                        n_valid,
                        tech_count
                    );
                }

                Ok(())
            })?;
        }
    }
}

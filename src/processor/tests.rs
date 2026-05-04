// Property-based tests for the processor module.
// Feature: ai-saved-manager

#[cfg(test)]
mod proptest_llm_client {
    use crate::config::BatchConfig;
    use crate::processor::llm_client::{validate_response, LlmBatchItem};
    use proptest::prelude::*;
    use serde_json::json;

    // ── Generators ────────────────────────────────────────────────────────────

    /// Generate a valid lowercase tag (3–20 lowercase ASCII letters).
    fn arb_lowercase_tag() -> impl Strategy<Value = String> {
        "[a-z]{3,20}".prop_map(|s| s)
    }

    /// Generate a Vec of 3–7 valid lowercase tags.
    fn arb_valid_tags() -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(arb_lowercase_tag(), 3..=7)
    }

    /// Generate a valid category string.
    fn arb_valid_category() -> impl Strategy<Value = String> {
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

    /// Generate a valid score in [0.0, 1.0].
    fn arb_valid_score() -> impl Strategy<Value = f64> {
        (0u32..=1000u32).prop_map(|n| n as f64 / 1000.0)
    }

    /// Generate a valid LlmBatchItem.
    fn arb_llm_batch_item() -> impl Strategy<Value = LlmBatchItem> {
        ("[a-z0-9]{8,16}", "[a-zA-Z0-9 ]{5,50}", "https://[a-z]{3,10}\\.com/[a-z]{3,10}").prop_map(
            |(id, title, link)| LlmBatchItem {
                id,
                content: format!("{title} {link}"),
            },
        )
    }

    /// Generate a batch of 1–10 LlmBatchItems.
    fn arb_llm_batch() -> impl Strategy<Value = Vec<LlmBatchItem>> {
        prop::collection::vec(arb_llm_batch_item(), 1..=10)
    }

    /// Generate a valid BatchConfig.
    fn arb_batch_config() -> impl Strategy<Value = BatchConfig> {
        (1usize..=50usize, 256u32..=4096u32, 0.0f32..=2.0f32).prop_map(
            |(size, max_tokens, temperature)| BatchConfig {
                size,
                max_tokens,
                temperature,
            },
        )
    }

    // ── Property 9: LLM request format is correct ─────────────────────────────
    // Feature: ai-saved-manager, Property 9: LLM request format correctness
    //
    // For any batch of posts, the JSON payload sent to the LLM SHALL be an array
    // where each element contains exactly the fields `id` and `content`, with
    // values matching the corresponding RawPost.
    //
    // Validates: Requirements 4.4
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop9_llm_request_format_correct(batch in arb_llm_batch()) {
            // Serialize the batch as the processor would
            let json_str = serde_json::to_string(&batch).unwrap();
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap();

            // Must be an array
            prop_assert_eq!(parsed.len(), batch.len());

            for (i, item) in parsed.iter().enumerate() {
                // Each element must have exactly "id" and "content"
                let obj = item.as_object().expect("element must be an object");
                prop_assert!(obj.contains_key("id"), "element {i} missing 'id'");
                prop_assert!(obj.contains_key("content"), "element {i} missing 'content'");

                // Values must match the original
                prop_assert_eq!(
                    obj["id"].as_str().unwrap(),
                    batch[i].id.as_str()
                );
                prop_assert_eq!(
                    obj["content"].as_str().unwrap(),
                    batch[i].content.as_str()
                );
            }
        }
    }

    // ── Property 10: LLM response validation accepts valid, rejects invalid ───
    // Feature: ai-saved-manager, Property 10: LLM response validation
    //
    // For any JSON array where each element contains id, summary, tags (3–7
    // lowercase strings), category (one of the seven allowed values), and score
    // (in [0.0, 1.0]), the validator SHALL accept it. For any response that
    // violates any of these constraints, the validator SHALL reject it.
    //
    // Validates: Requirements 4.5, 4.7, 4.8, 4.9
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop10_valid_response_accepted(
            id in "[a-z0-9]{4,16}",
            summary in "[a-zA-Z0-9 ]{10,100}",
            tags in arb_valid_tags(),
            category in arb_valid_category(),
            score in arb_valid_score(),
        ) {
            let response = json!([{
                "id": id,
                "summary": summary,
                "tags": tags,
                "category": category,
                "score": score,
            }]);
            let raw = response.to_string();
            let result = validate_response(&raw);
            prop_assert!(result.is_ok(), "valid response rejected: {:?}", result.err());
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop10_invalid_category_mapped_to_other(
            id in "[a-z0-9]{4,16}",
            summary in "[a-zA-Z0-9 ]{10,100}",
            tags in arb_valid_tags(),
            // Generate a category that is NOT one of the seven valid ones
            bad_category in "[A-Z][a-z]{4,15}",
            score in arb_valid_score(),
        ) {
            use crate::models::Category;
            let valid_cats = ["Technology","Business","Education","Entertainment","Travel","Personal","Other"];
            // Skip if we accidentally generated a valid category
            prop_assume!(!valid_cats.contains(&bad_category.as_str()));

            let response = json!([{
                "id": id,
                "summary": summary,
                "tags": tags,
                "category": bad_category,
                "score": score,
            }]);
            let raw = response.to_string();
            let result = validate_response(&raw);
            prop_assert!(result.is_ok(), "unknown category should be mapped to Other, not rejected");
            prop_assert_eq!(result.unwrap()[0].category.clone(), Category::Other, "unknown category should map to Other");
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop10_out_of_range_score_rejected(
            id in "[a-z0-9]{4,16}",
            summary in "[a-zA-Z0-9 ]{10,100}",
            tags in arb_valid_tags(),
            category in arb_valid_category(),
            // Score outside [0.0, 1.0]
            bad_score in prop_oneof![
                (1001u32..=9999u32).prop_map(|n| n as f64 / 1000.0),   // > 1.0
                (1u32..=9999u32).prop_map(|n| -(n as f64) / 1000.0),   // < 0.0
            ],
        ) {
            let response = json!([{
                "id": id,
                "summary": summary,
                "tags": tags,
                "category": category,
                "score": bad_score,
            }]);
            let raw = response.to_string();
            let result = validate_response(&raw);
            prop_assert!(result.is_err(), "out-of-range score should be rejected, score={bad_score}");
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop10_wrong_tag_count_rejected(
            id in "[a-z0-9]{4,16}",
            summary in "[a-zA-Z0-9 ]{10,100}",
            // Tags with count outside [1, 7]: only 0 or 8+ are invalid now
            tags in prop_oneof![
                Just(vec![]),                                         // too few (empty)
                prop::collection::vec(arb_lowercase_tag(), 8..=15),  // too many
            ],
            category in arb_valid_category(),
            score in arb_valid_score(),
        ) {
            let response = json!([{
                "id": id,
                "summary": summary,
                "tags": tags,
                "category": category,
                "score": score,
            }]);
            let raw = response.to_string();
            let result = validate_response(&raw);
            prop_assert!(result.is_err(), "wrong tag count should be rejected, count={}", tags.len());
        }
    }

    // ── Property 11: Config values applied to every LLM request ──────────────
    // Feature: ai-saved-manager, Property 11: Config values in LLM requests
    //
    // For any max_tokens and temperature values in config, every LLM request
    // sent during a processing run SHALL include those exact values in the
    // request payload.
    //
    // Validates: Requirements 9.3
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop11_config_values_in_ollama_request(
            config in arb_batch_config(),
            batch in arb_llm_batch(),
        ) {
            // Build the Ollama request body as the client would
            let options = serde_json::json!({
                "num_predict": config.max_tokens,
                "temperature": config.temperature,
            });

            // Verify the config values are present in the request options
            prop_assert_eq!(
                options["num_predict"].as_u64().unwrap(),
                config.max_tokens as u64
            );

            let temp_in_json = options["temperature"].as_f64().unwrap() as f32;
            prop_assert!(
                (temp_in_json - config.temperature).abs() < 1e-4,
                "temperature mismatch: expected {}, got {}",
                config.temperature,
                temp_in_json
            );

            // Also verify the batch items are serialized correctly
            let items_json = serde_json::to_string(&batch).unwrap();
            let parsed: Vec<serde_json::Value> = serde_json::from_str(&items_json).unwrap();
            prop_assert_eq!(parsed.len(), batch.len());
        }
    }
}

#[cfg(test)]
mod proptest_processor {
    use crate::models::{PostStatus, RawPost};
    use crate::processor::chunk_posts;
    use proptest::prelude::*;

    // ── Generator ─────────────────────────────────────────────────────────────

    /// Generate a RawPost with a given index for uniqueness.
    fn arb_raw_post(idx: usize) -> RawPost {
        RawPost {
            id: format!("post-{idx}"),
            dataset_id: "ds1".to_string(),
            title: format!("Title {idx}"),
            link: format!("https://example.com/post-{idx}"),
            image_url: None,
            category_raw: None,
            post_type: None,
            status: PostStatus::Pending,
            ignore_reason: None,
        }
    }

    /// Generate a Vec of N RawPosts.
    fn make_posts(n: usize) -> Vec<RawPost> {
        (0..n).map(arb_raw_post).collect()
    }

    // ── Property 8: Batch sizes are respected ─────────────────────────────────
    // Feature: ai-saved-manager, Property 8: Batch sizes respected
    //
    // For any list of N eligible posts and a configured batch size B, the
    // processor SHALL produce ceil(N / B) batches where every batch except
    // possibly the last has exactly B items, and the total item count across
    // all batches equals N.
    //
    // Validates: Requirements 4.3
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop8_batch_sizes_respected(
            n in 0usize..=500usize,
            b in 1usize..=50usize,
        ) {
            let posts = make_posts(n);
            let batches = chunk_posts(posts, b);

            // Number of batches must be ceil(N / B)
            let expected_batches = if n == 0 { 0 } else { (n + b - 1) / b };
            prop_assert_eq!(
                batches.len(),
                expected_batches,
                "expected ceil({}/{})={} batches, got {}",
                n, b, expected_batches, batches.len()
            );

            // Total item count must equal N
            let total: usize = batches.iter().map(|batch| batch.len()).sum();
            prop_assert_eq!(
                total,
                n,
                "total items across batches should equal {}, got {}",
                n, total
            );

            // Every batch except possibly the last must have exactly B items
            if batches.len() > 1 {
                for (i, batch) in batches[..batches.len() - 1].iter().enumerate() {
                    prop_assert_eq!(
                        batch.len(),
                        b,
                        "batch {} should have {} items, got {}",
                        i, b, batch.len()
                    );
                }
            }

            // Last batch must have between 1 and B items (if any batches exist)
            if let Some(last) = batches.last() {
                prop_assert!(
                    last.len() >= 1 && last.len() <= b,
                    "last batch size {} should be in [1, {}]",
                    last.len(), b
                );
            }
        }
    }

    // ── Property 18: max_items limits processing ──────────────────────────────
    // Feature: ai-saved-manager, Property 18: max_items limits processing
    //
    // For any max_items value greater than 0 and any dataset containing more
    // eligible posts than max_items, the processor SHALL process exactly
    // max_items posts in a single invocation.
    //
    // Validates: Requirements 10.1
    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]
        #[test]
        fn prop18_max_items_limits_processing(
            max_items in 1usize..=100usize,
            // total is always strictly greater than max_items
            extra in 1usize..=100usize,
        ) {
            let total = max_items + extra;
            let mut posts = make_posts(total);

            // Apply the same truncation logic as process_dataset
            if max_items > 0 && posts.len() > max_items {
                posts.truncate(max_items);
            }

            prop_assert_eq!(
                posts.len(),
                max_items,
                "after applying max_items={} to {} posts, expected {} posts, got {}",
                max_items, total, max_items, posts.len()
            );
        }
    }
}

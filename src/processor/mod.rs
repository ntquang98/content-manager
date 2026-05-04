pub mod filter;
pub mod llm_client;
#[cfg(test)]
mod tests;

use crate::config::AppConfig;
use crate::models::{PostStatus, ProcessedPost};
use crate::storage::Storage;
use filter::{ContentFilter, FilterResult};
use llm_client::{LlmBatchItem, LlmClient};

pub struct ProcessStats {
    pub processed: usize,
    pub skipped: usize,
    pub ignored: usize,
}

/// Chunk a vector of items into batches of `batch_size`.
/// The last batch may be smaller than `batch_size`.
pub fn chunk_posts<T: Clone>(posts: Vec<T>, batch_size: usize) -> Vec<Vec<T>> {
    if batch_size == 0 {
        return vec![];
    }
    posts.chunks(batch_size).map(|c| c.to_vec()).collect()
}

pub async fn process_dataset(
    dataset_id: &str,
    config: &AppConfig,
    storage: &Storage,
    llm: &dyn LlmClient,
) -> anyhow::Result<ProcessStats> {
    tracing::info!("Starting processing for dataset '{dataset_id}'");

    let mut posts = storage.get_unprocessed_posts(dataset_id).await?;
    tracing::info!(
        "Loaded {} unprocessed posts for dataset '{dataset_id}'",
        posts.len()
    );

    // `get_unprocessed_posts` already implements skip_existing=true behaviour
    // (it only returns posts with no analysis record). When skip_existing=false,
    // re-processing already-analysed posts is not yet supported.
    if !config.processing.skip_existing {
        tracing::debug!(
            "skip_existing=false: re-processing already-analyzed posts is not yet supported; \
             processing only unprocessed posts"
        );
    }

    // Apply max_items limit
    if config.processing.max_items > 0 && posts.len() > config.processing.max_items {
        tracing::info!(
            "Applying max_items limit: truncating from {} to {}",
            posts.len(),
            config.processing.max_items
        );
        posts.truncate(config.processing.max_items);
    }

    let mut stats = ProcessStats {
        processed: 0,
        skipped: 0,
        ignored: 0,
    };

    // Filter posts
    let mut eligible = Vec::new();
    for post in &posts {
        match ContentFilter::check(post, config.processing.min_content_length) {
            FilterResult::Pass => eligible.push(post.clone()),
            FilterResult::Ignore { reason } => {
                tracing::debug!(
                    "Post '{}' ignored: {reason}",
                    post.id
                );
                storage
                    .update_post_status(&post.id, PostStatus::Ignored, Some(&reason))
                    .await?;
                stats.ignored += 1;
            }
        }
    }

    tracing::info!(
        "{} posts eligible for LLM analysis, {} filtered out",
        eligible.len(),
        stats.ignored
    );

    // Process in batches
    let batch_size = config.llm.batch.size;
    let batches = chunk_posts(eligible, batch_size);
    let total_batches = batches.len();

    for (batch_idx, chunk) in batches.into_iter().enumerate() {
        tracing::info!(
            "Processing batch {}/{total_batches} with {} posts",
            batch_idx + 1,
            chunk.len()
        );

        let items: Vec<LlmBatchItem> = chunk
            .iter()
            .map(|p| LlmBatchItem {
                id: p.id.clone(),
                content: format!("{} {}", p.title, p.link),
            })
            .collect();

        // Build a set of valid IDs for this batch so we can reject any
        // hallucinated IDs the LLM returns that don't exist in the posts table.
        let valid_ids: std::collections::HashSet<&str> =
            items.iter().map(|i| i.id.as_str()).collect();

        match llm.analyze_batch(&items, &config.llm.batch).await {
            Ok(analyses) => {
                tracing::debug!(
                    "Batch {}/{total_batches}: received {} analyses",
                    batch_idx + 1,
                    analyses.len()
                );
                for analysis in analyses {
                    // Guard: skip any ID the LLM hallucinated or mangled —
                    // inserting an unknown post_id would fail the FK constraint.
                    if !valid_ids.contains(analysis.id.as_str()) {
                        tracing::warn!(
                            "Batch {}/{total_batches}: LLM returned unknown id '{}', skipping",
                            batch_idx + 1,
                            analysis.id
                        );
                        continue;
                    }
                    let processed = ProcessedPost {
                        post_id: analysis.id.clone(),
                        summary: analysis.summary,
                        tags: analysis.tags,
                        category_ai: analysis.category,
                        score: analysis.score,
                        processed_at: chrono::Utc::now().to_rfc3339(),
                    };
                    storage.insert_analysis(&processed).await?;
                    storage
                        .update_post_status(&analysis.id, PostStatus::Valid, None)
                        .await?;
                    stats.processed += 1;
                }
            }
            Err(e) => {
                tracing::error!(
                    "LLM batch {}/{total_batches} failed: {e}",
                    batch_idx + 1
                );
                for post in &chunk {
                    storage
                        .update_post_status(&post.id, PostStatus::Ignored, Some("llm_failure"))
                        .await?;
                    stats.ignored += 1;
                }
            }
        }
    }

    tracing::info!(
        "Processing complete for dataset '{dataset_id}': processed={}, skipped={}, ignored={}",
        stats.processed,
        stats.skipped,
        stats.ignored
    );

    Ok(stats)
}

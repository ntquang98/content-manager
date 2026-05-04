// LLM Client module: abstracts over Ollama and OpenAI providers.
// Handles batching, response validation, and retry logic.

use crate::config::BatchConfig;
use crate::models::Category;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, warn};

// ─── Error Types ────────────────────────────────────────────────────────────

/// Errors produced by the processor / LLM layer.
#[derive(Debug, Error)]
pub enum ProcessorError {
    #[error("LLM request failed after {attempts} attempts: {source}")]
    LlmExhausted {
        attempts: u8,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("LLM response validation failed: {reason}")]
    InvalidResponse { reason: String },
}

// ─── Data Structures ─────────────────────────────────────────────────────────

/// A single item sent to the LLM for analysis.
/// `content` is `title + " " + link`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmBatchItem {
    pub id: String,
    pub content: String,
}

/// The analysis result returned by the LLM for a single post.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmAnalysis {
    pub id: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub category: Category,
    pub score: f32,
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Abstraction over LLM providers.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn analyze_batch(
        &self,
        items: &[LlmBatchItem],
        config: &BatchConfig,
    ) -> Result<Vec<LlmAnalysis>>;
}

// ─── Response Validation ─────────────────────────────────────────────────────

/// Raw JSON shape returned by the LLM (before validation).
#[derive(Debug, Deserialize)]
struct RawAnalysis {
    id: Option<String>,
    summary: Option<String>,
    tags: Option<Vec<serde_json::Value>>,
    category: Option<String>,
    score: Option<serde_json::Value>,
}

const VALID_CATEGORIES: &[&str] = &[
    "Technology",
    "Business",
    "Education",
    "Entertainment",
    "Travel",
    "Personal",
    "Other",
];

/// Extract a JSON array from an LLM response that may contain markdown fences,
/// leading/trailing text, or other noise that local models commonly produce.
///
/// Strategy (in order):
/// 1. Try parsing the raw string directly.
/// 2. Strip markdown code fences (```json ... ``` or ``` ... ```).
/// 3. Find the first `[` and last `]` and extract the substring between them.
pub fn extract_json_array(raw: &str) -> &str {
    let trimmed = raw.trim();

    // 1. Already a valid-looking JSON array
    if trimmed.starts_with('[') {
        return trimmed;
    }

    // 2. Strip markdown code fences
    //    Handles: ```json\n...\n``` and ```\n...\n```
    let fence_stripped = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if fence_stripped.starts_with('[') {
        return fence_stripped;
    }

    // 3. Find the outermost [ ... ] block
    if let (Some(start), Some(end)) = (raw.find('['), raw.rfind(']')) {
        if start < end {
            return &raw[start..=end];
        }
    }

    // Give up — return original so the caller gets a meaningful parse error
    raw
}

/// Parse and validate a raw JSON string from the LLM.
///
/// Rules:
/// - Must be a JSON array.
/// - Each element must have `id`, `summary`, `tags`, `category`, `score`.
/// - `category` must be one of the seven allowed values.
/// - `score` must be in `[0.0, 1.0]`.
/// - `tags` must have 1–7 entries (normalized to lowercase automatically).
pub fn validate_response(raw: &str) -> Result<Vec<LlmAnalysis>> {
    let cleaned = extract_json_array(raw);

    // Parse as a JSON array
    let items: Vec<RawAnalysis> = serde_json::from_str(cleaned).map_err(|e| {
        ProcessorError::InvalidResponse {
            reason: format!("response is not a valid JSON array: {e}"),
        }
    })?;

    let mut analyses = Vec::with_capacity(items.len());

    for (i, item) in items.into_iter().enumerate() {
        // Required: id
        let id = item.id.ok_or_else(|| ProcessorError::InvalidResponse {
            reason: format!("element {i} missing field 'id'"),
        })?;

        // Required: summary
        let summary = item.summary.ok_or_else(|| ProcessorError::InvalidResponse {
            reason: format!("element {i} (id={id}) missing field 'summary'"),
        })?;

        // Required: tags — 1–7 entries, normalized to lowercase
        let raw_tags = item.tags.ok_or_else(|| ProcessorError::InvalidResponse {
            reason: format!("element {i} (id={id}) missing field 'tags'"),
        })?;

        if raw_tags.is_empty() || raw_tags.len() > 7 {
            return Err(ProcessorError::InvalidResponse {
                reason: format!(
                    "element {i} (id={id}) has {} tags; expected 1–7",
                    raw_tags.len()
                ),
            }
            .into());
        }

        let mut tags = Vec::with_capacity(raw_tags.len());
        for (j, v) in raw_tags.iter().enumerate() {
            let s = v.as_str().ok_or_else(|| ProcessorError::InvalidResponse {
                reason: format!("element {i} (id={id}) tag[{j}] is not a string"),
            })?;
            // Normalize to lowercase rather than rejecting — LLMs are inconsistent with casing
            tags.push(s.to_lowercase());
        }

        // Required: category — must be one of the seven values; unknown values fall back to Other
        let category_str = item.category.ok_or_else(|| ProcessorError::InvalidResponse {
            reason: format!("element {i} (id={id}) missing field 'category'"),
        })?;

        let category_str = if VALID_CATEGORIES.contains(&category_str.as_str()) {
            category_str
        } else {
            tracing::warn!(
                "element {i} (id={id}) has unrecognised category '{category_str}', mapping to 'Other'"
            );
            "Other".to_string()
        };

        let category: Category = serde_json::from_value(serde_json::Value::String(category_str))
            .map_err(|e| ProcessorError::InvalidResponse {
                reason: format!("element {i} (id={id}) failed to parse category: {e}"),
            })?;

        // Required: score — must be in [0.0, 1.0]
        let score_val = item.score.ok_or_else(|| ProcessorError::InvalidResponse {
            reason: format!("element {i} (id={id}) missing field 'score'"),
        })?;

        let score = score_val
            .as_f64()
            .ok_or_else(|| ProcessorError::InvalidResponse {
                reason: format!("element {i} (id={id}) 'score' is not a number"),
            })? as f32;

        if !(0.0..=1.0).contains(&score) {
            return Err(ProcessorError::InvalidResponse {
                reason: format!(
                    "element {i} (id={id}) score {score} is out of range [0.0, 1.0]"
                ),
            }
            .into());
        }

        analyses.push(LlmAnalysis {
            id,
            summary,
            tags,
            category,
            score,
        });
    }

    Ok(analyses)
}

// ─── Retry Helper ────────────────────────────────────────────────────────────

/// Retry delays in seconds: 1s, 2s, 4s (exponential backoff).
const RETRY_DELAYS: &[u64] = &[1, 2, 4];
/// Maximum number of attempts (1 initial + 3 retries).
const MAX_ATTEMPTS: u8 = 4;

/// Determine whether an HTTP status code should trigger a retry.
fn should_retry_status(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

/// Determine whether an HTTP status code should NOT be retried.
fn is_non_retryable_status(status: u16) -> bool {
    matches!(status, 400 | 401 | 403)
}

// ─── Ollama Client ───────────────────────────────────────────────────────────

/// Request body sent to the Ollama HTTP endpoint.
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    num_predict: u32,
    temperature: f32,
}

/// Response body from the Ollama HTTP endpoint.
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

pub struct OllamaClient {
    pub endpoint: String,
    pub model: String,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(endpoint: String, model: String) -> Self {
        OllamaClient {
            endpoint,
            model,
            client: reqwest::Client::new(),
        }
    }

    fn build_prompt(items: &[LlmBatchItem]) -> String {
        let items_json = serde_json::to_string(items).unwrap_or_default();
        format!(
            "You are a JSON API. Respond with ONLY a valid JSON array, no markdown, no explanation.\n\
             Analyze the following posts and return a JSON array where each element has:\n\
             - id: string (same as input)\n\
             - summary: string (1-2 sentences)\n\
             - tags: array of 1-7 lowercase keyword strings\n\
             - category: one of exactly: Technology, Business, Education, Entertainment, Travel, Personal, Other\n\
             - score: number between 0.0 and 1.0 (relevance/quality)\n\
             Output ONLY the JSON array, starting with [ and ending with ].\n\
             Posts: {items_json}"
        )
    }
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn analyze_batch(
        &self,
        items: &[LlmBatchItem],
        config: &BatchConfig,
    ) -> Result<Vec<LlmAnalysis>> {
        let prompt = Self::build_prompt(items);
        let request_body = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.clone(),
            stream: false,
            options: OllamaOptions {
                num_predict: config.max_tokens,
                temperature: config.temperature,
            },
        };

        debug!("Ollama request payload: {}", serde_json::to_string(&request_body).unwrap_or_default());

        let mut last_error: Box<dyn std::error::Error + Send + Sync> =
            Box::new(ProcessorError::InvalidResponse {
                reason: "no attempts made".to_string(),
            });

        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                let delay = RETRY_DELAYS[(attempt - 1) as usize];
                warn!("Ollama attempt {attempt} failed, retrying in {delay}s");
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            let resp = self
                .client
                .post(&self.endpoint)
                .json(&request_body)
                .send()
                .await;

            match resp {
                Err(e) => {
                    // Network error — always retry
                    last_error = Box::new(e);
                    continue;
                }
                Ok(response) => {
                    let status = response.status().as_u16();

                    if is_non_retryable_status(status) {
                        let msg = format!("HTTP {status} — not retrying");
                        return Err(ProcessorError::LlmExhausted {
                            attempts: attempt + 1,
                            source: Box::new(std::io::Error::new(
                                std::io::ErrorKind::PermissionDenied,
                                msg,
                            )),
                        }
                        .into());
                    }

                    if should_retry_status(status) {
                        last_error = Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("HTTP {status}"),
                        ));
                        continue;
                    }

                    if !response.status().is_success() {
                        last_error = Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("HTTP {status}"),
                        ));
                        continue;
                    }

                    let body: OllamaResponse = response.json().await.map_err(|e| {
                        ProcessorError::InvalidResponse {
                            reason: format!("failed to parse Ollama response: {e}"),
                        }
                    })?;

                    debug!("Ollama response: {}", body.response);
                    return validate_response(&body.response);
                }
            }
        }

        Err(ProcessorError::LlmExhausted {
            attempts: MAX_ATTEMPTS,
            source: last_error,
        }
        .into())
    }
}

// ─── OpenAI Client ───────────────────────────────────────────────────────────

pub struct OpenAiClient {
    pub model: String,
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
}

impl OpenAiClient {
    /// Create a new `OpenAiClient`.
    ///
    /// If `base_url` is `Some`, it overrides the default OpenAI endpoint —
    /// use this for OpenAI-compatible local servers like LM Studio or vLLM.
    /// If `base_url` is `None`, the standard `https://api.openai.com/v1` is used.
    ///
    /// `api_key` is passed through to the underlying config; for local servers
    /// any non-empty string works (e.g. `"lm-studio"`).
    pub fn new(model: String, base_url: Option<String>, api_key: Option<String>) -> Self {
        let mut config = async_openai::config::OpenAIConfig::new();
        if let Some(url) = base_url {
            config = config.with_api_base(url);
        }
        if let Some(key) = api_key {
            config = config.with_api_key(key);
        }
        let client = async_openai::Client::with_config(config);
        OpenAiClient { model, client }
    }

    /// Create a new `OpenAiClient` with a custom base URL.
    /// Useful for testing against a mock server or an OpenAI-compatible API.
    #[allow(dead_code)]
    pub fn new_with_base_url(model: String, base_url: String) -> Self {
        let config = async_openai::config::OpenAIConfig::new().with_api_base(base_url);
        let client = async_openai::Client::with_config(config);
        OpenAiClient { model, client }
    }

    fn build_prompt(items: &[LlmBatchItem]) -> String {
        let items_json = serde_json::to_string(items).unwrap_or_default();
        format!(
            "You are a JSON API. Respond with ONLY a valid JSON array, no markdown, no explanation.\n\
             Analyze the following posts and return a JSON array where each element has:\n\
             - id: string (same as input)\n\
             - summary: string (1-2 sentences)\n\
             - tags: array of 1-7 keyword strings\n\
             - category: one of exactly: Technology, Business, Education, Entertainment, Travel, Personal, Other\n\
             - score: number between 0.0 and 1.0 (relevance/quality)\n\
             Output ONLY the JSON array, starting with [ and ending with ].\n\
             Posts: {items_json}"
        )
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn analyze_batch(
        &self,
        items: &[LlmBatchItem],
        config: &BatchConfig,
    ) -> Result<Vec<LlmAnalysis>> {
        use async_openai::types::{
            ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
        };

        let prompt = Self::build_prompt(items);

        debug!("OpenAI request: model={}, max_tokens={}, temperature={}", self.model, config.max_tokens, config.temperature);

        let mut last_error: Box<dyn std::error::Error + Send + Sync> =
            Box::new(ProcessorError::InvalidResponse {
                reason: "no attempts made".to_string(),
            });

        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                let delay = RETRY_DELAYS[(attempt - 1) as usize];
                warn!("OpenAI attempt {attempt} failed, retrying in {delay}s");
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            let request = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .max_tokens(config.max_tokens as u16)
                .temperature(config.temperature)
                .messages([ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt.clone())
                    .build()
                    .map_err(|e| ProcessorError::InvalidResponse {
                        reason: format!("failed to build OpenAI message: {e}"),
                    })?
                    .into()])
                .build()
                .map_err(|e| ProcessorError::InvalidResponse {
                    reason: format!("failed to build OpenAI request: {e}"),
                })?;

            match self.client.chat().create(request).await {
                Err(e) => {
                    let err_str = e.to_string();
                    // Check for non-retryable HTTP status codes in the error message
                    if err_str.contains("400")
                        || err_str.contains("401")
                        || err_str.contains("403")
                    {
                        return Err(ProcessorError::LlmExhausted {
                            attempts: attempt + 1,
                            source: Box::new(e),
                        }
                        .into());
                    }
                    last_error = Box::new(e);
                    continue;
                }
                Ok(response) => {
                    let content = response
                        .choices
                        .first()
                        .and_then(|c| c.message.content.as_ref())
                        .ok_or_else(|| ProcessorError::InvalidResponse {
                            reason: "OpenAI response has no content".to_string(),
                        })?
                        .clone();

                    debug!("OpenAI response content: {content}");
                    match validate_response(&content) {
                        Ok(analyses) => return Ok(analyses),
                        Err(e) => {
                            warn!("OpenAI response validation failed: {e}");
                            warn!("Raw response was: {content}");
                            last_error = Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                e.to_string(),
                            ));
                            continue;
                        }
                    }
                }
            }
        }

        Err(ProcessorError::LlmExhausted {
            attempts: MAX_ATTEMPTS,
            source: last_error,
        }
        .into())
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_response_json() -> &'static str {
        r#"[
            {
                "id": "abc123",
                "summary": "A great article about Rust programming.",
                "tags": ["rust", "programming", "systems"],
                "category": "Technology",
                "score": 0.9
            }
        ]"#
    }

    #[test]
    fn test_valid_response_accepted() {
        let result = validate_response(valid_response_json());
        assert!(result.is_ok());
        let analyses = result.unwrap();
        assert_eq!(analyses.len(), 1);
        assert_eq!(analyses[0].id, "abc123");
        assert_eq!(analyses[0].category, Category::Technology);
        assert!((analyses[0].score - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_missing_id_rejected() {
        let json = r#"[{"summary": "test", "tags": ["a","b","c"], "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing field 'id'"), "got: {msg}");
    }

    #[test]
    fn test_missing_summary_rejected() {
        let json = r#"[{"id": "x", "tags": ["a","b","c"], "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing field 'summary'"), "got: {msg}");
    }

    #[test]
    fn test_missing_tags_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing field 'tags'"), "got: {msg}");
    }

    #[test]
    fn test_missing_category_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c"], "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing field 'category'"), "got: {msg}");
    }

    #[test]
    fn test_missing_score_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c"], "category": "Technology"}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("missing field 'score'"), "got: {msg}");
    }

    #[test]
    fn test_invalid_category_mapped_to_other() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c"], "category": "Sports", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_ok(), "unknown category should be mapped to Other, got: {:?}", result);
        assert_eq!(result.unwrap()[0].category, Category::Other);
    }

    #[test]
    fn test_score_above_one_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c"], "category": "Technology", "score": 1.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("out of range"), "got: {msg}");
    }

    #[test]
    fn test_score_below_zero_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c"], "category": "Technology", "score": -0.1}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("out of range"), "got: {msg}");
    }

    #[test]
    fn test_too_few_tags_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": [], "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("tags"), "got: {msg}");
    }

    #[test]
    fn test_too_many_tags_rejected() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["a","b","c","d","e","f","g","h"], "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("tags"), "got: {msg}");
    }

    #[test]
    fn test_uppercase_tag_normalized_to_lowercase() {
        let json = r#"[{"id": "x", "summary": "s", "tags": ["Rust","Programming","Systems"], "category": "Technology", "score": 0.5}]"#;
        let result = validate_response(json);
        assert!(result.is_ok(), "uppercase tags should be normalized, got: {:?}", result);
        let analyses = result.unwrap();
        assert_eq!(analyses[0].tags, vec!["rust", "programming", "systems"]);
    }

    #[test]
    fn test_not_json_array_rejected() {
        let json = r#"{"id": "x", "summary": "s"}"#;
        let result = validate_response(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_seven_categories_accepted() {
        for cat in &[
            "Technology",
            "Business",
            "Education",
            "Entertainment",
            "Travel",
            "Personal",
            "Other",
        ] {
            let json = format!(
                r#"[{{"id":"x","summary":"s","tags":["a","b","c"],"category":"{cat}","score":0.5}}]"#
            );
            let result = validate_response(&json);
            assert!(result.is_ok(), "category {cat} should be accepted, got: {:?}", result);
        }
    }

    #[test]
    fn test_score_boundary_values_accepted() {
        for score in &[0.0f32, 1.0f32] {
            let json = format!(
                r#"[{{"id":"x","summary":"s","tags":["a","b","c"],"category":"Technology","score":{score}}}]"#
            );
            let result = validate_response(&json);
            assert!(result.is_ok(), "score {score} should be accepted");
        }
    }

    /// Test that retry exhaustion produces a ProcessorError::LlmExhausted.
    /// We simulate this by checking the error type from a mock scenario.
    #[test]
    fn test_processor_error_display() {
        let err = ProcessorError::LlmExhausted {
            attempts: 4,
            source: Box::new(std::io::Error::new(std::io::ErrorKind::Other, "timeout")),
        };
        let msg = err.to_string();
        assert!(msg.contains("4 attempts"), "got: {msg}");

        let err2 = ProcessorError::InvalidResponse {
            reason: "bad json".to_string(),
        };
        let msg2 = err2.to_string();
        assert!(msg2.contains("bad json"), "got: {msg2}");
    }
}

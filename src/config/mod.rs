use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found at {path}: {source}")]
    NotFound {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse config.toml: {0}")]
    ParseError(#[from] toml::de::Error),
    #[error("OpenAI provider requires OPENAI_API_KEY environment variable")]
    MissingApiKey,
    #[error("invalid logging level '{value}'; valid values are: error, warn, info, debug, trace")]
    InvalidLogLevel { value: String },
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    Ollama,
    OpenAi,
}

impl Default for LlmProvider {
    fn default() -> Self {
        LlmProvider::Ollama
    }
}

fn default_endpoint() -> String {
    "http://localhost:11434/api/generate".to_string()
}

fn default_model() -> String {
    "llama3".to_string()
}

fn default_batch_size() -> usize {
    10
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f32 {
    0.3
}

fn default_skip_existing() -> bool {
    true
}

fn default_min_content_length() -> usize {
    20
}

fn default_max_items() -> usize {
    0
}

fn default_output_dir() -> String {
    "output".to_string()
}

fn default_db_path() -> String {
    "data/ai-saved-manager.db".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct BatchConfig {
    #[serde(default = "default_batch_size")]
    pub size: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            size: default_batch_size(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlmConfig {
    #[serde(default)]
    pub provider: LlmProvider,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub batch: BatchConfig,
}

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            provider: LlmProvider::default(),
            endpoint: default_endpoint(),
            model: default_model(),
            batch: BatchConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProcessingConfig {
    #[serde(default = "default_skip_existing")]
    pub skip_existing: bool,
    #[serde(default = "default_min_content_length")]
    pub min_content_length: usize,
    #[serde(default = "default_max_items")]
    pub max_items: usize,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        ProcessingConfig {
            skip_existing: default_skip_existing(),
            min_content_length: default_min_content_length(),
            max_items: default_max_items(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    #[serde(default = "default_output_dir")]
    pub dir: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            dir: default_output_dir(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct StorageConfig {
    #[serde(default = "default_db_path")]
    pub path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig {
            path: default_db_path(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            level: default_log_level(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            llm: LlmConfig::default(),
            processing: ProcessingConfig::default(),
            output: OutputConfig::default(),
            storage: StorageConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

const VALID_LOG_LEVELS: &[&str] = &["error", "warn", "info", "debug", "trace"];

impl AppConfig {
    pub fn load(path: &Path) -> Result<AppConfig, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::NotFound {
            path: path.display().to_string(),
            source: e,
        })?;

        let config: AppConfig = toml::from_str(&content)?;

        // Validate log level
        if !VALID_LOG_LEVELS.contains(&config.logging.level.as_str()) {
            return Err(ConfigError::InvalidLogLevel {
                value: config.logging.level.clone(),
            });
        }

        // Validate OpenAI API key — only required when using the real OpenAI endpoint.
        // Local OpenAI-compatible servers (LM Studio, vLLM, etc.) don't need a real key.
        if config.llm.provider == LlmProvider::OpenAi
            && !config.llm.endpoint.contains("localhost")
            && !config.llm.endpoint.contains("127.0.0.1")
            && std::env::var("OPENAI_API_KEY").is_err()
        {
            return Err(ConfigError::MissingApiKey);
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_config(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // Feature: ai-saved-manager, Property 1: Config defaults are applied for all absent optional fields
    proptest! {
        #[test]
        fn prop_config_defaults_for_absent_fields(
            include_batch_size in proptest::bool::ANY,
            include_max_tokens in proptest::bool::ANY,
            include_temperature in proptest::bool::ANY,
            include_skip_existing in proptest::bool::ANY,
            include_min_content_length in proptest::bool::ANY,
            include_max_items in proptest::bool::ANY,
            include_output_dir in proptest::bool::ANY,
            include_storage_path in proptest::bool::ANY,
            include_log_level in proptest::bool::ANY,
        ) {
            // Build a TOML string that only includes the fields selected by the booleans.
            // All non-included fields should fall back to their documented defaults.
            let mut toml = String::new();

            // [llm.batch] section
            let has_batch_section = include_batch_size || include_max_tokens || include_temperature;
            if has_batch_section {
                toml.push_str("[llm.batch]\n");
                if include_batch_size {
                    toml.push_str("size = 10\n");
                }
                if include_max_tokens {
                    toml.push_str("max_tokens = 2048\n");
                }
                if include_temperature {
                    toml.push_str("temperature = 0.3\n");
                }
            }

            // [processing] section
            let has_processing_section = include_skip_existing || include_min_content_length || include_max_items;
            if has_processing_section {
                toml.push_str("[processing]\n");
                if include_skip_existing {
                    toml.push_str("skip_existing = true\n");
                }
                if include_min_content_length {
                    toml.push_str("min_content_length = 20\n");
                }
                if include_max_items {
                    toml.push_str("max_items = 0\n");
                }
            }

            // [output] section
            if include_output_dir {
                toml.push_str("[output]\n");
                toml.push_str("dir = \"output\"\n");
            }

            // [storage] section
            if include_storage_path {
                toml.push_str("[storage]\n");
                toml.push_str("path = \"data/ai-saved-manager.db\"\n");
            }

            // [logging] section — always use a valid level to avoid InvalidLogLevel error
            if include_log_level {
                toml.push_str("[logging]\n");
                toml.push_str("level = \"info\"\n");
            }

            let f = write_config(&toml);
            let cfg = AppConfig::load(f.path()).expect("config should load successfully");

            // Verify every omitted field has its documented default value
            if !include_batch_size {
                prop_assert_eq!(cfg.llm.batch.size, 10, "default batch size should be 10");
            }
            if !include_max_tokens {
                prop_assert_eq!(cfg.llm.batch.max_tokens, 2048, "default max_tokens should be 2048");
            }
            if !include_temperature {
                prop_assert!(
                    (cfg.llm.batch.temperature - 0.3_f32).abs() < 1e-6,
                    "default temperature should be 0.3, got {}",
                    cfg.llm.batch.temperature
                );
            }
            if !include_skip_existing {
                prop_assert_eq!(cfg.processing.skip_existing, true, "default skip_existing should be true");
            }
            if !include_min_content_length {
                prop_assert_eq!(cfg.processing.min_content_length, 20, "default min_content_length should be 20");
            }
            if !include_max_items {
                prop_assert_eq!(cfg.processing.max_items, 0, "default max_items should be 0");
            }
            if !include_output_dir {
                prop_assert_eq!(cfg.output.dir, "output", "default output dir should be \"output\"");
            }
            if !include_storage_path {
                prop_assert_eq!(cfg.storage.path, "data/ai-saved-manager.db", "default storage path should be \"data/ai-saved-manager.db\"");
            }
            if !include_log_level {
                prop_assert_eq!(cfg.logging.level, "info", "default log level should be \"info\"");
            }
        }
    }

    #[test]
    fn test_valid_config() {
        let f = write_config(
            r#"
[llm]
provider = "ollama"
endpoint = "http://localhost:11434/api/generate"
model = "llama3"

[llm.batch]
size = 10
max_tokens = 2048
temperature = 0.3

[processing]
skip_existing = true
min_content_length = 20
max_items = 0

[output]
dir = "output"

[storage]
path = "data/test.db"

[logging]
level = "info"
"#,
        );
        let cfg = AppConfig::load(f.path()).unwrap();
        assert_eq!(cfg.llm.batch.size, 10);
        assert_eq!(cfg.logging.level, "info");
    }

    #[test]
    fn test_missing_file() {
        let result = AppConfig::load(Path::new("/nonexistent/config.toml"));
        assert!(matches!(result, Err(ConfigError::NotFound { .. })));
    }

    #[test]
    fn test_malformed_toml() {
        let f = write_config("not valid toml ][");
        let result = AppConfig::load(f.path());
        assert!(matches!(result, Err(ConfigError::ParseError(_))));
    }

    #[test]
    fn test_invalid_log_level() {
        let f = write_config("[logging]\nlevel = \"verbose\"");
        let result = AppConfig::load(f.path());
        assert!(matches!(result, Err(ConfigError::InvalidLogLevel { .. })));
    }

    #[test]
    fn test_defaults_applied() {
        let f = write_config("");
        let cfg = AppConfig::load(f.path()).unwrap();
        assert_eq!(cfg.llm.batch.size, 10);
        assert_eq!(cfg.llm.batch.max_tokens, 2048);
        assert!((cfg.llm.batch.temperature - 0.3).abs() < 1e-6);
        assert_eq!(cfg.processing.min_content_length, 20);
        assert_eq!(cfg.logging.level, "info");
    }

    #[test]
    fn test_openai_missing_api_key() {
        // API key is only required when using the real OpenAI endpoint (not localhost).
        // A config with provider=openai but no endpoint defaults to the Ollama endpoint
        // (localhost), so no key is required.
        let f = write_config("[llm]\nprovider = \"openai\"");
        std::env::remove_var("OPENAI_API_KEY");
        let result = AppConfig::load(f.path());
        // Default endpoint is localhost — no key required, should succeed
        assert!(
            result.is_ok(),
            "local endpoint should not require OPENAI_API_KEY, got: {:?}", result
        );
    }

    #[test]
    fn test_openai_real_endpoint_requires_api_key() {
        let f = write_config(
            "[llm]\nprovider = \"openai\"\nendpoint = \"https://api.openai.com/v1\""
        );
        std::env::remove_var("OPENAI_API_KEY");
        let result = AppConfig::load(f.path());
        assert!(
            matches!(result, Err(ConfigError::MissingApiKey)),
            "real OpenAI endpoint should require OPENAI_API_KEY, got: {:?}", result
        );
    }
}

// Command handler implementations for each CLI subcommand.

use crate::config::AppConfig;
use crate::exporter::{Exporter, JsonExporter, SqliteExporter};
use crate::importer::{FacebookImporter, Importer};
use crate::processor::{self, llm_client::{OllamaClient, OpenAiClient}};
use crate::storage::Storage;
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};

// ── Import ────────────────────────────────────────────────────────────────────

/// Handle the `import` subcommand.
///
/// 1. Load config from `config.toml`.
/// 2. Open storage at `config.storage.path`.
/// 3. Initialise the schema.
/// 4. Find or create the dataset by name.
/// 5. Resolve the importer by `--source`.
/// 6. Run the import and print stats.
pub async fn run_import(source: &str, dataset: &str, file: &Path) -> Result<()> {
    let config = AppConfig::load(Path::new("config.toml"))
        .context("failed to load config.toml")?;

    let storage = Storage::open(Path::new(&config.storage.path))
        .await
        .context("failed to open storage")?;

    storage.init_schema().await.context("failed to initialise schema")?;

    // Find or create the dataset
    let ds = match storage.find_dataset_by_name(dataset).await? {
        Some(ds) => ds,
        None => storage
            .create_dataset(dataset, source)
            .await
            .context("failed to create dataset")?,
    };

    tracing::info!("Importing into dataset '{}' (id={})", ds.name, ds.id);

    // Resolve importer by source
    let stats = match source {
        "facebook" => {
            let importer = FacebookImporter;
            importer.import(file, &ds.id, &storage).await?
        }
        other => {
            return Err(anyhow!("unknown import source '{}'; supported: facebook", other));
        }
    };

    println!(
        "Imported: parsed={}, inserted={}, skipped={}",
        stats.parsed, stats.inserted, stats.skipped
    );

    Ok(())
}

// ── Process ───────────────────────────────────────────────────────────────────

/// Handle the `process` subcommand.
///
/// 1. Load config.
/// 2. Open storage.
/// 3. Build the LLM client from `config.llm.provider`.
/// 4. Run `process_dataset` and print stats.
pub async fn run_process(dataset: &str) -> Result<()> {
    let config = AppConfig::load(Path::new("config.toml"))
        .context("failed to load config.toml")?;

    let storage = Storage::open(Path::new(&config.storage.path))
        .await
        .context("failed to open storage")?;

    // Find the dataset
    let ds = storage
        .find_dataset_by_name(dataset)
        .await?
        .ok_or_else(|| anyhow!("dataset '{}' not found", dataset))?;

    tracing::info!("Processing dataset '{}' (id={})", ds.name, ds.id);

    // Build LLM client from provider config
    use crate::config::LlmProvider;
    let stats = match config.llm.provider {
        LlmProvider::Ollama => {
            let llm = OllamaClient::new(
                config.llm.endpoint.clone(),
                config.llm.model.clone(),
            );
            processor::process_dataset(&ds.id, &config, &storage, &llm).await?
        }
        LlmProvider::OpenAi => {
            // For OpenAI-compatible local servers (LM Studio, vLLM, etc.),
            // `config.llm.endpoint` holds the base URL (e.g. http://localhost:1234/v1).
            // For real OpenAI, leave endpoint at its default and set OPENAI_API_KEY.
            let base_url = if config.llm.endpoint.contains("openai.com") {
                None
            } else {
                Some(config.llm.endpoint.clone())
            };
            let api_key = std::env::var("OPENAI_API_KEY").ok();
            let llm = OpenAiClient::new(
                config.llm.model.clone(),
                base_url,
                api_key,
            );
            processor::process_dataset(&ds.id, &config, &storage, &llm).await?
        }
    };

    println!(
        "Processed: processed={}, skipped={}, ignored={}",
        stats.processed, stats.skipped, stats.ignored
    );

    Ok(())
}

// ── Export ────────────────────────────────────────────────────────────────────

/// Handle the `export` subcommand.
///
/// 1. Load config.
/// 2. Open storage.
/// 3. Find the dataset and fetch export items.
/// 4. Resolve the exporter by `--format`.
/// 5. Run the export and print the output path and item count.
pub async fn run_export(dataset: &str, format: &str, output: &PathBuf) -> Result<()> {
    let config = AppConfig::load(Path::new("config.toml"))
        .context("failed to load config.toml")?;

    let storage = Storage::open(Path::new(&config.storage.path))
        .await
        .context("failed to open storage")?;

    // Find the dataset
    let ds = storage
        .find_dataset_by_name(dataset)
        .await?
        .ok_or_else(|| anyhow!("dataset '{}' not found", dataset))?;

    tracing::info!("Exporting dataset '{}' (id={})", ds.name, ds.id);

    // If the output path has no parent directory component, use config.output.dir
    // as the default output directory.
    let output = if output.parent().map_or(true, |p| p == Path::new("")) {
        tracing::debug!(
            "No directory in output path '{}'; using config output.dir='{}'",
            output.display(),
            config.output.dir
        );
        PathBuf::from(&config.output.dir).join(output)
    } else {
        output.clone()
    };

    let items = storage
        .get_export_items(&ds.id)
        .await
        .context("failed to fetch export items")?;

    let count = match format {
        "json" => {
            let exporter = JsonExporter;
            exporter.export(&items, &output).await?
        }
        "sqlite" => {
            let exporter = SqliteExporter;
            exporter.export(&items, &output).await?
        }
        other => {
            return Err(anyhow!(
                "unknown export format '{}'; supported: json, sqlite",
                other
            ));
        }
    };

    println!("Exported {} items to {}", count, output.display());

    Ok(())
}

// ── Datasets ──────────────────────────────────────────────────────────────────

/// Handle the `datasets` subcommand.
///
/// Lists all datasets, printing name, source, and creation timestamp.
pub async fn run_datasets() -> Result<()> {
    let config = AppConfig::load(Path::new("config.toml"))
        .context("failed to load config.toml")?;

    let storage = Storage::open(Path::new(&config.storage.path))
        .await
        .context("failed to open storage")?;

    let datasets = storage.list_datasets().await.context("failed to list datasets")?;

    if datasets.is_empty() {
        println!("No datasets found.");
    } else {
        for ds in &datasets {
            println!("{} ({}) created {}", ds.name, ds.source, ds.created_at);
        }
    }

    Ok(())
}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Handle the `stats` subcommand.
///
/// Prints all five metrics for the specified dataset.
/// Exits with an error if the dataset is not found.
pub async fn run_stats(dataset: &str) -> Result<()> {
    let config = AppConfig::load(Path::new("config.toml"))
        .context("failed to load config.toml")?;

    let storage = Storage::open(Path::new(&config.storage.path))
        .await
        .context("failed to open storage")?;

    // Find the dataset — user error if not found
    let ds = storage
        .find_dataset_by_name(dataset)
        .await?
        .ok_or_else(|| anyhow!("dataset '{}' not found", dataset))?;

    let stats = storage
        .get_dataset_stats(&ds.id)
        .await
        .context("failed to retrieve dataset stats")?;

    println!("Dataset: {}", ds.name);
    println!("  Total:       {}", stats.total);
    println!("  Valid:       {}", stats.valid);
    println!("  Ignored:     {}", stats.ignored);
    println!("  Unprocessed: {}", stats.unprocessed);
    println!("  Category distribution:");
    if stats.category_distribution.is_empty() {
        println!("    (none)");
    } else {
        for (category, count) in &stats.category_distribution {
            println!("    {}: {}", category, count);
        }
    }

    Ok(())
}

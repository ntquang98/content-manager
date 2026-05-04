mod cli;
mod config;
mod exporter;
mod importer;
mod models;
mod processor;
mod storage;

use clap::Parser;
use cli::{Cli, Commands};
use config::{AppConfig, ConfigError};
use std::path::Path;
use std::process;

#[tokio::main]
async fn main() {
    let cli_args = Cli::parse();

    // Load config first so we can use the configured log level.
    // If config loading fails we still need to emit the error, so fall back to
    // a default "info" subscriber for that one message.
    let config = match AppConfig::load(Path::new("config.toml")) {
        Ok(cfg) => cfg,
        Err(e) => {
            // Initialise a minimal subscriber so the error is visible.
            let _ = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::INFO)
                .try_init();
            eprintln!("Error: {e}");
            // Config errors are user errors → exit code 1.
            process::exit(1);
        }
    };

    // Initialise tracing with the level from config.
    let level = cli::parse_log_level(&config.logging.level);
    tracing_subscriber::fmt()
        .with_max_level(level)
        .init();

    // Dispatch to the appropriate command handler.
    let result = dispatch(&cli_args.command).await;

    match result {
        Ok(()) => {}
        Err(e) => {
            // Classify the error to choose the exit code.
            let exit_code = classify_error(&e);
            eprintln!("Error: {e:#}");
            process::exit(exit_code);
        }
    }
}

/// Dispatch to the correct command handler.
async fn dispatch(command: &Commands) -> anyhow::Result<()> {
    match command {
        Commands::Import { source, dataset, file } => {
            cli::commands::run_import(source, dataset, file).await
        }
        Commands::Process { dataset } => {
            cli::commands::run_process(dataset).await
        }
        Commands::Export { dataset, format, output } => {
            cli::commands::run_export(dataset, format, output).await
        }
        Commands::Datasets => {
            cli::commands::run_datasets().await
        }
        Commands::Stats { dataset } => {
            cli::commands::run_stats(dataset).await
        }
    }
}

/// Map an error to an exit code.
///
/// - Exit code 1: user errors (bad config, dataset not found, unknown source/format).
/// - Exit code 2: internal errors (unexpected storage or LLM failures).
fn classify_error(err: &anyhow::Error) -> i32 {
    // Check for ConfigError variants
    if let Some(config_err) = err.downcast_ref::<ConfigError>() {
        match config_err {
            ConfigError::NotFound { .. }
            | ConfigError::ParseError(_)
            | ConfigError::MissingApiKey
            | ConfigError::InvalidLogLevel { .. } => return 1,
        }
    }

    // Check for StorageError::DatasetNotFound
    if let Some(storage_err) = err.downcast_ref::<crate::storage::StorageError>() {
        if matches!(storage_err, crate::storage::StorageError::DatasetNotFound { .. }) {
            return 1;
        }
    }

    // "dataset not found" / user-facing messages from anyhow::anyhow!
    let msg = err.to_string();
    if msg.contains("not found")
        || msg.contains("unknown import source")
        || msg.contains("unknown export format")
    {
        return 1;
    }

    // Everything else is an internal error.
    2
}

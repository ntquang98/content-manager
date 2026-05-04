// CLI module — Task 10
// Defines the Cli struct, Commands enum, and all command handlers.

pub mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Convert a validated log-level string to a `tracing::Level`.
///
/// The config module guarantees the string is one of the five valid values.
/// The fallback to INFO is a defensive default for any unexpected input.
pub fn parse_log_level(level: &str) -> tracing::Level {
    match level {
        "error" => tracing::Level::ERROR,
        "warn"  => tracing::Level::WARN,
        "info"  => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _       => tracing::Level::INFO,
    }
}

#[derive(Parser)]
#[command(name = "ai-saved-manager", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Import a raw export file into a dataset
    Import {
        #[arg(long)]
        source: String,
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        file: PathBuf,
    },
    /// Run LLM analysis on unprocessed posts
    Process {
        #[arg(long)]
        dataset: String,
    },
    /// Export a processed dataset to a file
    Export {
        #[arg(long)]
        dataset: String,
        /// Output format: "json" or "sqlite"
        #[arg(long)]
        format: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// List all datasets
    Datasets,
    /// Show statistics for a dataset
    Stats {
        #[arg(long)]
        dataset: String,
    },
}

#[cfg(test)]
mod tests;

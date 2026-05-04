use crate::models::ExportItem;
use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

#[cfg(test)]
mod tests;

/// Trait for exporting a slice of [`ExportItem`]s to an output file.
///
/// Returns the number of items written.
#[async_trait]
pub trait Exporter: Send {
    async fn export(&self, items: &[ExportItem], output: &Path) -> Result<usize>;
}

// ── JSON Exporter ─────────────────────────────────────────────────────────────

/// Exports items as a pretty-printed JSON file: `{ "items": [...] }`.
pub struct JsonExporter;

#[async_trait]
impl Exporter for JsonExporter {
    async fn export(&self, items: &[ExportItem], output: &Path) -> Result<usize> {
        // 9.4 — create output directory if it does not exist
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        tokio::fs::create_dir_all(parent).await?;

        // 9.5 — warn if the file already exists
        if tokio::fs::metadata(output).await.is_ok() {
            tracing::warn!(
                path = %output.display(),
                "output file already exists, overwriting"
            );
        }

        // Write `{ "items": [...] }` using serde_json::to_writer_pretty
        let file = std::fs::File::create(output)?;
        let wrapper = serde_json::json!({ "items": items });
        serde_json::to_writer_pretty(file, &wrapper)?;

        Ok(items.len())
    }
}

// ── SQLite Exporter ───────────────────────────────────────────────────────────

/// Exports items into a new SQLite file, inserting all rows in a single transaction.
pub struct SqliteExporter;

#[async_trait]
impl Exporter for SqliteExporter {
    async fn export(&self, items: &[ExportItem], output: &Path) -> Result<usize> {
        // 9.4 — create output directory if it does not exist
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        tokio::fs::create_dir_all(parent).await?;

        // 9.5 — warn if the file already exists; remove it so rusqlite creates a fresh database
        if tokio::fs::metadata(output).await.is_ok() {
            tracing::warn!(
                path = %output.display(),
                "output file already exists, overwriting"
            );
            tokio::fs::remove_file(output).await?;
        }

        let conn = rusqlite::Connection::open(output)?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS items (
                title       TEXT NOT NULL,
                summary     TEXT NOT NULL,
                tags        TEXT NOT NULL,
                category_ai TEXT NOT NULL,
                link        TEXT NOT NULL,
                image_url   TEXT,
                score       REAL NOT NULL
            );",
        )?;

        // Insert all items in a single transaction
        let tx = conn.unchecked_transaction()?;
        for item in items {
            let tags_json = serde_json::to_string(&item.tags)?;
            tx.execute(
                "INSERT INTO items (title, summary, tags, category_ai, link, image_url, score) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    item.title,
                    item.summary,
                    tags_json,
                    item.category_ai,
                    item.link,
                    item.image_url,
                    item.score
                ],
            )?;
        }
        tx.commit()?;

        Ok(items.len())
    }
}

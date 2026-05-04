use crate::models::{Dataset, ExportItem, PostStatus, ProcessedPost, RawPost};

#[cfg(test)]
mod tests;
use std::path::Path;
use thiserror::Error;
use tokio_rusqlite::Connection;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("database error: {0}")]
    Rusqlite(#[from] rusqlite::Error),
    #[error("tokio-rusqlite error: {0}")]
    TokioRusqlite(#[from] tokio_rusqlite::Error),
    #[error("dataset '{name}' not found")]
    DatasetNotFound { name: String },
    #[error("dataset name '{name}' already exists")]
    DuplicateDataset { name: String },
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct Storage {
    conn: Connection,
}

#[derive(Debug)]
pub struct DatasetStats {
    pub total: usize,
    pub valid: usize,
    pub ignored: usize,
    pub unprocessed: usize,
    pub category_distribution: Vec<(String, usize)>,
}

impl Storage {
    pub async fn open(path: &Path) -> Result<Self, StorageError> {
        let conn = Connection::open(path).await?;
        Ok(Storage { conn })
    }

    #[allow(dead_code)]
    pub async fn open_in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory().await?;
        Ok(Storage { conn })
    }

    pub async fn init_schema(&self) -> Result<(), StorageError> {
        self.conn
            .call(|conn| {
                conn.execute_batch(
                    "PRAGMA foreign_keys = ON;

                    CREATE TABLE IF NOT EXISTS datasets (
                        id         TEXT PRIMARY KEY,
                        name       TEXT UNIQUE NOT NULL,
                        source     TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS posts (
                        id            TEXT PRIMARY KEY,
                        dataset_id    TEXT NOT NULL REFERENCES datasets(id),
                        title         TEXT NOT NULL,
                        link          TEXT NOT NULL,
                        image_url     TEXT,
                        category_raw  TEXT,
                        post_type     TEXT,
                        status        TEXT NOT NULL DEFAULT 'pending',
                        ignore_reason TEXT
                    );

                    CREATE TABLE IF NOT EXISTS analysis (
                        post_id      TEXT PRIMARY KEY REFERENCES posts(id),
                        summary      TEXT NOT NULL,
                        tags         TEXT NOT NULL,
                        category_ai  TEXT NOT NULL,
                        score        REAL NOT NULL,
                        processed_at TEXT NOT NULL
                    );",
                )
                .map_err(tokio_rusqlite::Error::from)
            })
            .await?;
        Ok(())
    }

    pub async fn create_dataset(&self, name: &str, source: &str) -> Result<Dataset, StorageError> {
        let id = uuid::Uuid::new_v4().to_string();
        let created_at = chrono::Utc::now().to_rfc3339();
        let name = name.to_string();
        let source = source.to_string();
        let id_clone = id.clone();
        let name_clone = name.clone();
        let source_clone = source.clone();
        let created_at_clone = created_at.clone();

        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO datasets (id, name, source, created_at) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![id_clone, name_clone, source_clone, created_at_clone],
                )
                .map_err(tokio_rusqlite::Error::from)
            })
            .await
            .map_err(|e| {
                // Check if it's a unique constraint violation
                if e.to_string().contains("UNIQUE constraint failed") {
                    StorageError::DuplicateDataset { name: name.clone() }
                } else {
                    StorageError::TokioRusqlite(e)
                }
            })?;

        Ok(Dataset {
            id,
            name,
            source,
            created_at,
        })
    }

    pub async fn find_dataset_by_name(
        &self,
        name: &str,
    ) -> Result<Option<Dataset>, StorageError> {
        let name = name.to_string();
        let result = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, name, source, created_at FROM datasets WHERE name = ?1",
                )?;
                let mut rows = stmt.query(rusqlite::params![name])?;
                if let Some(row) = rows.next()? {
                    Ok(Some((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    )))
                } else {
                    Ok(None)
                }
            })
            .await?;

        Ok(result.map(|(id, name, source, created_at)| Dataset {
            id,
            name,
            source,
            created_at,
        }))
    }

    pub async fn list_datasets(&self) -> Result<Vec<Dataset>, StorageError> {
        let rows = self
            .conn
            .call(|conn| {
                let mut stmt =
                    conn.prepare("SELECT id, name, source, created_at FROM datasets ORDER BY created_at")?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;

        Ok(rows
            .into_iter()
            .map(|(id, name, source, created_at)| Dataset {
                id,
                name,
                source,
                created_at,
            })
            .collect())
    }

    pub async fn insert_post(&self, post: &RawPost) -> Result<bool, StorageError> {
        let post = post.clone();
        let inserted = self
            .conn
            .call(move |conn| {
                let result = conn.execute(
                    "INSERT OR IGNORE INTO posts (id, dataset_id, title, link, image_url, category_raw, post_type, status, ignore_reason)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    rusqlite::params![
                        post.id,
                        post.dataset_id,
                        post.title,
                        post.link,
                        post.image_url,
                        post.category_raw,
                        post.post_type,
                        post.status.to_string(),
                        post.ignore_reason,
                    ],
                )?;
                Ok(result > 0)
            })
            .await?;
        Ok(inserted)
    }

    pub async fn get_unprocessed_posts(
        &self,
        dataset_id: &str,
    ) -> Result<Vec<RawPost>, StorageError> {
        let dataset_id = dataset_id.to_string();
        let rows = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT p.id, p.dataset_id, p.title, p.link, p.image_url, p.category_raw, p.post_type, p.status, p.ignore_reason
                     FROM posts p
                     LEFT JOIN analysis a ON p.id = a.post_id
                     WHERE p.dataset_id = ?1 AND a.post_id IS NULL AND p.status = 'pending'",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![dataset_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, Option<String>>(4)?,
                            row.get::<_, Option<String>>(5)?,
                            row.get::<_, Option<String>>(6)?,
                            row.get::<_, String>(7)?,
                            row.get::<_, Option<String>>(8)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;

        Ok(rows
            .into_iter()
            .map(
                |(id, dataset_id, title, link, image_url, category_raw, post_type, status, ignore_reason)| {
                    let status = match status.as_str() {
                        "valid" => PostStatus::Valid,
                        "ignored" => PostStatus::Ignored,
                        _ => PostStatus::Pending,
                    };
                    RawPost {
                        id,
                        dataset_id,
                        title,
                        link,
                        image_url,
                        category_raw,
                        post_type,
                        status,
                        ignore_reason,
                    }
                },
            )
            .collect())
    }

    pub async fn update_post_status(
        &self,
        id: &str,
        status: PostStatus,
        reason: Option<&str>,
    ) -> Result<(), StorageError> {
        let id = id.to_string();
        let status_str = status.to_string();
        let reason = reason.map(|s| s.to_string());
        self.conn
            .call(move |conn| {
                conn.execute(
                    "UPDATE posts SET status = ?1, ignore_reason = ?2 WHERE id = ?3",
                    rusqlite::params![status_str, reason, id],
                )?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    pub async fn insert_analysis(&self, analysis: &ProcessedPost) -> Result<(), StorageError> {
        let analysis = analysis.clone();
        let tags_json = serde_json::to_string(&analysis.tags)?;
        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT OR REPLACE INTO analysis (post_id, summary, tags, category_ai, score, processed_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        analysis.post_id,
                        analysis.summary,
                        tags_json,
                        analysis.category_ai.to_string(),
                        analysis.score,
                        analysis.processed_at,
                    ],
                )?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    pub async fn get_dataset_stats(
        &self,
        dataset_id: &str,
    ) -> Result<DatasetStats, StorageError> {
        let dataset_id = dataset_id.to_string();
        let dataset_id_for_err = dataset_id.clone();
        let stats = self
            .conn
            .call(move |conn| {
                let total: usize = conn.query_row(
                    "SELECT COUNT(*) FROM posts WHERE dataset_id = ?1",
                    rusqlite::params![dataset_id],
                    |row| row.get(0),
                )?;

                // If no posts found, check whether the dataset itself exists.
                if total == 0 {
                    let dataset_exists: usize = conn.query_row(
                        "SELECT COUNT(*) FROM datasets WHERE id = ?1",
                        rusqlite::params![dataset_id],
                        |row| row.get(0),
                    )?;
                    if dataset_exists == 0 {
                        return Err(tokio_rusqlite::Error::Other(
                            format!("dataset '{}' not found", dataset_id).into(),
                        ));
                    }
                }

                let valid: usize = conn.query_row(
                    "SELECT COUNT(*) FROM posts WHERE dataset_id = ?1 AND status = 'valid'",
                    rusqlite::params![dataset_id],
                    |row| row.get(0),
                )?;
                let ignored: usize = conn.query_row(
                    "SELECT COUNT(*) FROM posts WHERE dataset_id = ?1 AND status = 'ignored'",
                    rusqlite::params![dataset_id],
                    |row| row.get(0),
                )?;
                let unprocessed: usize = conn.query_row(
                    "SELECT COUNT(*) FROM posts p LEFT JOIN analysis a ON p.id = a.post_id WHERE p.dataset_id = ?1 AND a.post_id IS NULL",
                    rusqlite::params![dataset_id],
                    |row| row.get(0),
                )?;

                let mut stmt = conn.prepare(
                    "SELECT a.category_ai, COUNT(*) FROM posts p JOIN analysis a ON p.id = a.post_id WHERE p.dataset_id = ?1 GROUP BY a.category_ai",
                )?;
                let category_distribution = stmt
                    .query_map(rusqlite::params![dataset_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;

                Ok((total, valid, ignored, unprocessed, category_distribution))
            })
            .await
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("not found") {
                    StorageError::DatasetNotFound { name: dataset_id_for_err }
                } else {
                    StorageError::TokioRusqlite(e)
                }
            })?;

        Ok(DatasetStats {
            total: stats.0,
            valid: stats.1,
            ignored: stats.2,
            unprocessed: stats.3,
            category_distribution: stats.4,
        })
    }

    pub async fn get_export_items(
        &self,
        dataset_id: &str,
    ) -> Result<Vec<ExportItem>, StorageError> {
        let dataset_id = dataset_id.to_string();
        let rows = self
            .conn
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT p.title, a.summary, a.tags, a.category_ai, p.link, p.image_url, a.score
                     FROM posts p
                     JOIN analysis a ON p.id = a.post_id
                     WHERE p.dataset_id = ?1",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![dataset_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, Option<String>>(5)?,
                            row.get::<_, f64>(6)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;

        let mut items = Vec::new();
        for (title, summary, tags_json, category_ai, link, image_url, score) in rows {
            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            items.push(ExportItem {
                title,
                summary,
                tags,
                category_ai,
                link,
                image_url,
                score: score as f32,
            });
        }
        Ok(items)
    }
}

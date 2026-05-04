use crate::models::{PostStatus, RawPost};
use crate::storage::Storage;

#[cfg(test)]
mod tests;
use serde::Deserialize;
use sha1::{Digest, Sha1};
use std::future::Future;
use std::path::Path;

#[derive(Debug)]
pub struct ImportStats {
    pub parsed: usize,
    pub inserted: usize,
    pub skipped: usize,
}

pub trait Importer: Send {
    fn import<'a>(
        &'a self,
        file: &'a Path,
        dataset_id: &'a str,
        storage: &'a Storage,
    ) -> impl Future<Output = anyhow::Result<ImportStats>> + Send + 'a;
}

/// A record from the J2Team Facebook saved posts export format.
#[derive(Debug, Deserialize)]
pub struct J2TeamRecord {
    #[serde(rename = "Title")]
    pub title: Option<String>,
    #[serde(rename = "Link")]
    pub link: Option<String>,
    #[serde(rename = "Image URL")]
    pub image_url: Option<String>,
    #[serde(rename = "Category")]
    pub category: Option<String>,
    #[serde(rename = "Type")]
    pub post_type: Option<String>,
}

/// Compute the SHA-1 hex digest of a string.
pub fn sha1_hex(input: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

pub struct Normalizer;

impl Normalizer {
    /// Convert a J2TeamRecord into a RawPost. Returns None if the link is empty or absent.
    pub fn normalize(record: J2TeamRecord, dataset_id: &str) -> Option<RawPost> {
        let link = record.link.filter(|l| !l.trim().is_empty())?;
        let id = sha1_hex(&link);
        Some(RawPost {
            id,
            dataset_id: dataset_id.to_string(),
            title: record.title.unwrap_or_default(),
            link,
            image_url: record.image_url,
            category_raw: record.category,
            post_type: record.post_type,
            status: PostStatus::Pending,
            ignore_reason: None,
        })
    }
}

pub struct FacebookImporter;

impl Importer for FacebookImporter {
    fn import<'a>(
        &'a self,
        file: &'a Path,
        dataset_id: &'a str,
        storage: &'a Storage,
    ) -> impl Future<Output = anyhow::Result<ImportStats>> + Send + 'a {
        async move {
            // Read the file as bytes and strip a UTF-8 BOM (EF BB BF) if present.
            // The J2Team Security extension exports UTF-8 with BOM, which
            // serde_json cannot parse ("expected value at line 1 column 1").
            let raw = std::fs::read(file)?;
            let bytes = raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&raw);

            let records: Vec<J2TeamRecord> = serde_json::from_slice(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON file: {}", e))?;

            let mut stats = ImportStats {
                parsed: 0,
                inserted: 0,
                skipped: 0,
            };

            for record in records {
                stats.parsed += 1;
                match Normalizer::normalize(record, dataset_id) {
                    None => {
                        stats.skipped += 1;
                    }
                    Some(post) => {
                        if storage.insert_post(&post).await? {
                            stats.inserted += 1;
                        } else {
                            stats.skipped += 1;
                        }
                    }
                }
            }

            Ok(stats)
        }
    }
}

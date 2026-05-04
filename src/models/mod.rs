use serde::{Deserialize, Serialize};
use std::fmt;

/// Status of a post in the processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PostStatus {
    Pending,
    Valid,
    Ignored,
}

impl fmt::Display for PostStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PostStatus::Pending => write!(f, "pending"),
            PostStatus::Valid => write!(f, "valid"),
            PostStatus::Ignored => write!(f, "ignored"),
        }
    }
}

/// AI-assigned category for a post.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum Category {
    Technology,
    Business,
    Education,
    Entertainment,
    Travel,
    Personal,
    Other,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Category::Technology => write!(f, "Technology"),
            Category::Business => write!(f, "Business"),
            Category::Education => write!(f, "Education"),
            Category::Entertainment => write!(f, "Entertainment"),
            Category::Travel => write!(f, "Travel"),
            Category::Personal => write!(f, "Personal"),
            Category::Other => write!(f, "Other"),
        }
    }
}

/// A normalized post record produced during the import phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawPost {
    pub id: String,
    pub dataset_id: String,
    pub title: String,
    pub link: String,
    pub image_url: Option<String>,
    pub category_raw: Option<String>,
    pub post_type: Option<String>,
    pub status: PostStatus,
    pub ignore_reason: Option<String>,
}

/// An analysis record produced by the LLM for a given RawPost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedPost {
    pub post_id: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub category_ai: Category,
    pub score: f32,
    pub processed_at: String,
}

/// A named, independent collection of imported posts.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub source: String,
    pub created_at: String,
}

/// An item ready for export, joining post and analysis data.
#[derive(Debug, Serialize)]
pub struct ExportItem {
    pub title: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub category_ai: String,
    pub link: String,
    pub image_url: Option<String>,
    pub score: f32,
}

use crate::models::RawPost;

/// Result of applying content filter rules to a post.
#[derive(Debug, PartialEq)]
pub enum FilterResult {
    Pass,
    Ignore { reason: String },
}

pub struct ContentFilter;

impl ContentFilter {
    /// Apply content quality rules to a post.
    pub fn check(post: &RawPost, min_content_length: usize) -> FilterResult {
        // Rule 1: empty or whitespace-only title
        if post.title.trim().is_empty() {
            return FilterResult::Ignore {
                reason: "empty_title".to_string(),
            };
        }

        // Rule 2: combined length below minimum
        if post.title.len() + post.link.len() < min_content_length {
            return FilterResult::Ignore {
                reason: "content_too_short".to_string(),
            };
        }

        FilterResult::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::PostStatus;

    fn make_post(title: &str, link: &str) -> RawPost {
        RawPost {
            id: "test-id".to_string(),
            dataset_id: "ds1".to_string(),
            title: title.to_string(),
            link: link.to_string(),
            image_url: None,
            category_raw: None,
            post_type: None,
            status: PostStatus::Pending,
            ignore_reason: None,
        }
    }

    #[test]
    fn test_empty_title_ignored() {
        let post = make_post("", "https://example.com/some-long-link");
        assert_eq!(
            ContentFilter::check(&post, 20),
            FilterResult::Ignore {
                reason: "empty_title".to_string()
            }
        );
    }

    #[test]
    fn test_whitespace_only_title_ignored() {
        let post = make_post("   \t\n  ", "https://example.com/some-long-link");
        assert_eq!(
            ContentFilter::check(&post, 20),
            FilterResult::Ignore {
                reason: "empty_title".to_string()
            }
        );
    }

    #[test]
    fn test_short_content_ignored() {
        let post = make_post("Hi", "http://x.co");
        // "Hi".len() + "http://x.co".len() = 2 + 11 = 13 < 20
        assert_eq!(
            ContentFilter::check(&post, 20),
            FilterResult::Ignore {
                reason: "content_too_short".to_string()
            }
        );
    }

    #[test]
    fn test_valid_post_passes() {
        let post = make_post("A good title", "https://example.com/article");
        assert_eq!(ContentFilter::check(&post, 20), FilterResult::Pass);
    }
}

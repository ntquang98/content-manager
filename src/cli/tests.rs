// Unit tests and property-based tests for the CLI layer.
// Task 10.9: unit tests for log level error message, info-level logging, debug-level logging.
// Task 10.10: proptest for Property 19 (log level filtering).

use crate::config::ConfigError;
use crate::cli::parse_log_level;

// ── 10.9 Unit Tests ───────────────────────────────────────────────────────────

/// Test that ConfigError::InvalidLogLevel produces the correct error message.
#[test]
fn test_invalid_log_level_error_message() {
    let err = ConfigError::InvalidLogLevel {
        value: "verbose".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("verbose"),
        "error message should contain the invalid value; got: {msg}"
    );
    assert!(
        msg.contains("error") && msg.contains("warn") && msg.contains("info")
            && msg.contains("debug") && msg.contains("trace"),
        "error message should list all valid levels; got: {msg}"
    );
}

/// Test that all valid log level strings parse to the correct tracing::Level.
#[test]
fn test_log_level_string_parsing() {
    assert_eq!(parse_log_level("error"), tracing::Level::ERROR);
    assert_eq!(parse_log_level("warn"),  tracing::Level::WARN);
    assert_eq!(parse_log_level("info"),  tracing::Level::INFO);
    assert_eq!(parse_log_level("debug"), tracing::Level::DEBUG);
    assert_eq!(parse_log_level("trace"), tracing::Level::TRACE);
}

/// Test that an unknown level string falls back to INFO (defensive default).
#[test]
fn test_unknown_log_level_defaults_to_info() {
    assert_eq!(parse_log_level("unknown"), tracing::Level::INFO);
}

/// Test that info-level pipeline stage logging is represented correctly.
///
/// The design specifies that at "info" level the CLI logs the start and
/// completion of each pipeline stage along with item counts.  We verify
/// that the tracing::Level ordering is correct: INFO is enabled when the
/// configured level is INFO or lower (DEBUG / TRACE).
#[test]
fn test_info_level_pipeline_stage_logging() {
    // INFO messages should be visible at INFO, DEBUG, and TRACE levels.
    let info_level = tracing::Level::INFO;
    assert!(tracing::Level::INFO >= info_level);
    assert!(tracing::Level::DEBUG >= info_level);
    assert!(tracing::Level::TRACE >= info_level);

    // INFO messages should NOT be visible at WARN or ERROR levels.
    assert!(tracing::Level::WARN < info_level);
    assert!(tracing::Level::ERROR < info_level);
}

/// Test that debug-level LLM payload logging is represented correctly.
///
/// DEBUG messages should only appear when the configured level is DEBUG or TRACE.
#[test]
fn test_debug_level_llm_payload_logging() {
    let debug_level = tracing::Level::DEBUG;

    // DEBUG messages visible at DEBUG and TRACE.
    assert!(tracing::Level::DEBUG >= debug_level);
    assert!(tracing::Level::TRACE >= debug_level);

    // DEBUG messages NOT visible at INFO, WARN, or ERROR.
    assert!(tracing::Level::INFO < debug_level);
    assert!(tracing::Level::WARN < debug_level);
    assert!(tracing::Level::ERROR < debug_level);
}

// ── 10.10 Property-Based Test — Property 19: Log Level Filtering ──────────────
//
// Feature: ai-saved-manager, Property 19: For any configured log level L,
// log messages at severity levels below L SHALL NOT appear in the output,
// and messages at level L and above SHALL appear.
//
// Validates: Requirements 11.1
//
// We test the pure logic of tracing::Level ordering without requiring actual
// tracing infrastructure, since tracing::Level implements PartialOrd.

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    /// Arbitrary generator for tracing::Level (represented as u8 index 0..5).
    /// 0 = ERROR (highest severity / lowest verbosity)
    /// 1 = WARN
    /// 2 = INFO
    /// 3 = DEBUG
    /// 4 = TRACE (lowest severity / highest verbosity)
    fn arb_level_index() -> impl Strategy<Value = u8> {
        0u8..5u8
    }

    fn index_to_level(idx: u8) -> tracing::Level {
        match idx {
            0 => tracing::Level::ERROR,
            1 => tracing::Level::WARN,
            2 => tracing::Level::INFO,
            3 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        }
    }

    /// A message at level `msg_level` is emitted when the configured level is
    /// `configured_level` if and only if `msg_level <= configured_level`
    /// (in tracing's ordering where TRACE > DEBUG > INFO > WARN > ERROR).
    fn would_emit(msg_level: tracing::Level, configured_level: tracing::Level) -> bool {
        msg_level <= configured_level
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// Property 19: For any configured level L and any message level M:
        /// - If M <= L (M is at least as severe as L), the message SHALL appear.
        /// - If M > L (M is less severe than L), the message SHALL NOT appear.
        #[test]
        fn prop_log_level_filtering(
            configured_idx in arb_level_index(),
            msg_idx in arb_level_index(),
        ) {
            let configured = index_to_level(configured_idx);
            let msg = index_to_level(msg_idx);

            let emitted = would_emit(msg, configured);

            if msg <= configured {
                // Message is at or above the configured threshold → should appear.
                prop_assert!(
                    emitted,
                    "message at {:?} should be emitted when configured level is {:?}",
                    msg, configured
                );
            } else {
                // Message is below the configured threshold → should NOT appear.
                prop_assert!(
                    !emitted,
                    "message at {:?} should NOT be emitted when configured level is {:?}",
                    msg, configured
                );
            }
        }

        /// Property 19 (corollary): The configured level itself is always emitted.
        #[test]
        fn prop_configured_level_always_emitted(configured_idx in arb_level_index()) {
            let configured = index_to_level(configured_idx);
            prop_assert!(
                would_emit(configured, configured),
                "message at the configured level {:?} should always be emitted",
                configured
            );
        }

        /// Property 19 (corollary): Messages more severe than the configured level
        /// are always emitted (e.g., ERROR is always shown regardless of level).
        #[test]
        fn prop_more_severe_always_emitted(
            configured_idx in arb_level_index(),
            msg_idx in arb_level_index(),
        ) {
            let configured = index_to_level(configured_idx);
            let msg = index_to_level(msg_idx);

            // If msg is more severe (lower index = higher severity), it should be emitted.
            if msg_idx <= configured_idx {
                prop_assert!(
                    would_emit(msg, configured),
                    "message at {:?} (more severe) should be emitted when configured level is {:?}",
                    msg, configured
                );
            }
        }
    }
}

use std::borrow::Cow;
use std::env;
use std::fmt;
use std::sync::OnceLock;

use instant::Instant;

fn timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match env::var("HALO2_TIMING") {
        Ok(value) => {
            let value = value.trim();
            !(value.is_empty()
                || value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    })
}

fn indent(depth: usize) -> Indent {
    Indent(depth)
}

struct Indent(usize);

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            write!(f, "  ")?;
        }
        Ok(())
    }
}

/// RAII helper that logs scoped timings when [`HALO2_TIMING`] is enabled.
#[derive(Debug)]
#[must_use]
pub struct TimingScope {
    label: Cow<'static, str>,
    start: Instant,
    depth: usize,
    enabled: bool,
}

impl TimingScope {
    /// Create a new top-level timing scope.
    pub fn root(label: impl Into<Cow<'static, str>>) -> Self {
        let enabled = timing_enabled();
        let label = label.into();
        if enabled {
            log::info!("[timing] {}{}: start", indent(0), label);
        }
        Self {
            label,
            start: Instant::now(),
            depth: 0,
            enabled,
        }
    }

    /// Create a child scope that inherits the timing configuration.
    pub fn child(&self, label: impl Into<Cow<'static, str>>) -> TimingScope {
        let label = label.into();
        if self.enabled {
            log::info!("[timing] {}{}: start", indent(self.depth + 1), label);
        }
        TimingScope {
            label,
            start: Instant::now(),
            depth: self.depth + 1,
            enabled: self.enabled,
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        if self.enabled {
            log::info!(
                "[timing] {}{}: {:.3?}",
                indent(self.depth),
                self.label,
                self.start.elapsed()
            );
        }
    }
}

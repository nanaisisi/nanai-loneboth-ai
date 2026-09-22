//! Intel GNA (Gaussian and Neural Accelerator) module
//!
//! Provides hardware-accelerated low-power inference using Intel GNA via `nanai-burn-gna`.

use anyhow::{Result, anyhow};
use nanai_burn_gna::{GnaBackend, GnaDevice, GnaLibrary};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// GNA configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnaConfig {
    /// Target GNA device index (default: 0)
    pub device_index: u16,
    /// Fallback to CPU/GPU backend if GNA hardware or library is unavailable
    pub fallback_on_unavailable: bool,
    /// Enable profiling / performance metric tracking
    pub enable_metrics: bool,
}

impl Default for GnaConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            fallback_on_unavailable: true,
            enable_metrics: false,
        }
    }
}

/// GNA accelerator managing device detection, capability inspection, and execution
pub struct GnaAccelerator {
    config: GnaConfig,
    device: GnaDevice,
    available: bool,
}

impl GnaAccelerator {
    /// Create a new GNA accelerator with the given configuration
    pub fn new(config: GnaConfig) -> Result<Self> {
        let available = Self::check_availability();
        if !available && !config.fallback_on_unavailable {
            return Err(anyhow!(
                "Intel GNA device or runtime library is not available"
            ));
        }

        let device = GnaDevice::new(config.device_index);
        info!(
            device_index = config.device_index,
            available = available,
            "Initialized GnaAccelerator"
        );

        Ok(Self {
            config,
            device,
            available,
        })
    }

    /// Check whether GNA runtime library can be loaded and devices are accessible
    pub fn check_availability() -> bool {
        match GnaLibrary::load_default() {
            Ok(lib) => {
                let count = GnaDevice::get_count(&lib).unwrap_or(0);
                debug!("GNA library loaded successfully. Available device count: {count}");
                count > 0
            }
            Err(err) => {
                debug!("GNA library unavailable: {err}");
                false
            }
        }
    }

    /// Get the count of available GNA hardware devices
    pub fn device_count() -> usize {
        <GnaBackend as burn::tensor::backend::Backend>::device_count(0x474e)
    }

    /// Check if GNA is currently available
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Get underlying Burn GnaDevice
    pub fn device(&self) -> GnaDevice {
        self.device
    }

    /// Get current configuration
    pub fn config(&self) -> &GnaConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gna_config_default() {
        let config = GnaConfig::default();
        assert_eq!(config.device_index, 0);
        assert!(config.fallback_on_unavailable);
    }

    #[test]
    fn test_gna_accelerator_init() {
        let config = GnaConfig {
            device_index: 0,
            fallback_on_unavailable: true,
            enable_metrics: false,
        };
        let accelerator = GnaAccelerator::new(config).expect("GnaAccelerator init with fallback");
        assert_eq!(accelerator.device().index, 0);
    }

    #[test]
    fn test_gna_device_count() {
        let count = GnaAccelerator::device_count();
        let _ = count;
    }
}

//! Verification configuration definitions

use serde::{Deserialize, Serialize};

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable consistency verification
    pub consistency_enabled: bool,
    /// Enable behavioral validation
    pub behavioral_enabled: bool,
    /// Enable structural integrity checking
    pub structural_enabled: bool,
    /// Enable performance verification
    pub performance_enabled: bool,
    /// Verification thresholds
    pub thresholds: VerificationThresholds,
    /// Real-time verification
    pub real_time_enabled: bool,
}

/// Verification thresholds for different checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationThresholds {
    /// Consistency threshold
    pub consistency_threshold: f32,
    /// Behavioral coherence threshold
    pub behavioral_threshold: f32,
    /// Structural integrity threshold
    pub structural_threshold: f32,
    /// Performance threshold
    pub performance_threshold: f32,
    /// Overall system health threshold
    pub system_health_threshold: f32,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            consistency_enabled: true,
            behavioral_enabled: true,
            structural_enabled: true,
            performance_enabled: true,
            thresholds: VerificationThresholds {
                consistency_threshold: 0.8,
                behavioral_threshold: 0.7,
                structural_threshold: 0.85,
                performance_threshold: 0.75,
                system_health_threshold: 0.8,
            },
            real_time_enabled: false,
        }
    }
}

//! Inference types, records and metadata

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Inference performance metrics
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    /// Average inference latency
    pub average_latency: Duration,
    /// Total inferences performed
    pub total_inferences: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f32,
    /// Decision confidence distribution
    pub confidence_distribution: Vec<f32>,
    /// Adaptation frequency
    pub adaptation_frequency: f32,
    /// Start time for metrics collection
    pub start_time: Option<Instant>,
}

/// Inference result with confidence and metadata
#[derive(Debug, Clone)]
pub struct InferenceResult<B: Backend> {
    /// Predicted action
    pub action: Tensor<B, 2>,
    /// Confidence in prediction
    pub confidence: f32,
    /// Uncertainty estimate
    pub uncertainty: f32,
    /// Decision metadata
    pub metadata: InferenceMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    /// Inference method used
    pub method: String,
    /// Processing time
    pub processing_time: Duration,
    /// Cache hit indicator
    pub cache_hit: bool,
    /// Adaptation applied
    pub adaptation_applied: bool,
}

/// Decision record for temporal reasoning
#[derive(Debug, Clone)]
pub struct DecisionRecord<B: Backend> {
    /// Input state
    pub state: Tensor<B, 2>,
    /// Generated action
    pub action: Tensor<B, 2>,
    /// Decision confidence
    pub confidence: f32,
    /// Decision timestamp
    pub timestamp: Instant,
    /// Environmental context
    pub context: EnvironmentalContext,
}

/// Environmental context for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    /// Environmental state identifier
    pub state_id: String,
    /// Context features
    pub features: Vec<f32>,
    /// Temporal evolution rate
    pub evolution_rate: f32,
    /// Uncertainty level
    pub uncertainty: f32,
}

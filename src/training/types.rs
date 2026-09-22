//! Training types, metrics and events

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Training batch containing experience data
#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    /// State observations
    pub states: Tensor<B, 3>,
    /// Actions taken
    pub actions: Tensor<B, 3>,
    /// Rewards received
    pub rewards: Tensor<B, 2>,
    /// Next states
    pub next_states: Tensor<B, 3>,
    /// Episode termination flags
    pub done: Tensor<B, 2>,
    /// Temporal context
    pub temporal_context: Tensor<B, 3>,
}

/// Training result containing learned parameters and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Updated policy network weights
    pub policy_weights: Vec<f32>,
    /// Training loss progression
    pub loss_history: Vec<f32>,
    /// Accuracy metrics
    pub accuracy_history: Vec<f32>,
    /// Adaptation events during training
    pub adaptation_events: Vec<AdaptationEvent>,
    /// Training duration
    pub training_duration: Duration,
}

/// Training metrics collection and monitoring
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Loss values over time
    pub loss_history: Vec<f32>,
    /// Accuracy values over time
    pub accuracy_history: Vec<f32>,
    /// Learning rate over time
    pub lr_history: Vec<f64>,
    /// Adaptation frequency
    pub adaptation_frequency: Vec<f32>,
    /// Training start time
    pub start_time: Option<Instant>,
}

/// Adaptation event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Type of adaptation
    pub adaptation_type: AdaptationType,
    /// Magnitude of change
    pub magnitude: f32,
    /// Success indicator
    pub success: bool,
    /// Context information
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    Structural,
    Relational,
    Behavioral,
    Environmental,
    Combined,
}

/// Adaptation result information
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    pub magnitude: f32,
    pub success: bool,
}

/// Adaptation analysis result
#[derive(Debug, Clone)]
pub struct AdaptationAnalysis {
    pub structural_needs: f32,
    pub relational_needs: f32,
    pub priority: f32,
}

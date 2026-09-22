//! Types, data structures, and actions for adaptation

use crate::EnvironmentState;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Adaptation pattern for learning and reuse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationPattern<B: Backend> {
    /// Pattern identifier
    pub id: String,
    /// Environmental conditions when pattern was successful
    pub conditions: Vec<f32>,
    /// Adaptation actions taken
    pub actions: AdaptationActions,
    /// Success metrics
    pub success_metrics: SuccessMetrics,
    /// Learned parameters
    pub learned_parameters: Vec<f32>,
    /// Usage frequency
    pub usage_count: u64,
    /// Last used timestamp
    pub last_used: u64,
    /// Pattern effectiveness score
    pub effectiveness: f32,
    #[serde(skip)]
    pub _phantom: std::marker::PhantomData<B>,
}

/// Specific adaptation actions taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationActions {
    /// Structural modifications
    pub structural_changes: Vec<StructuralChange>,
    /// Relational modifications
    pub relational_changes: Vec<RelationalChange>,
    /// Parameter adjustments
    pub parameter_adjustments: Vec<ParameterAdjustment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralChange {
    /// Type of structural change
    pub change_type: StructuralChangeType,
    /// Target component
    pub target: String,
    /// Modification parameters
    pub parameters: Vec<f32>,
    /// Change magnitude
    pub magnitude: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralChangeType {
    /// Add new component
    Addition,
    /// Remove existing component
    Removal,
    /// Modify existing component
    Modification,
    /// Restructure component relationships
    Restructuring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalChange {
    /// Type of relational change
    pub change_type: RelationalChangeType,
    /// Source component
    pub source: String,
    /// Target component
    pub target: String,
    /// Relationship parameters
    pub parameters: Vec<f32>,
    /// Change strength
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationalChangeType {
    /// Strengthen relationship
    Strengthening,
    /// Weaken relationship
    Weakening,
    /// Create new relationship
    Creation,
    /// Remove relationship
    Removal,
    /// Modify relationship type
    TypeModification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterAdjustment {
    /// Parameter name
    pub parameter: String,
    /// Old value
    pub old_value: f32,
    /// New value
    pub new_value: f32,
    /// Adjustment reason
    pub reason: String,
}

/// Success metrics for adaptation evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetrics {
    /// Performance improvement
    pub performance_gain: f32,
    /// Adaptation speed
    pub adaptation_speed: Duration,
    /// Stability after adaptation
    pub stability_score: f32,
    /// Resource efficiency
    pub efficiency_score: f32,
    /// Overall success rating
    pub overall_rating: f32,
}

/// Adaptation failure record for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationFailure {
    /// Failure identifier
    pub id: String,
    /// Attempted adaptation
    pub attempted_adaptation: AdaptationActions,
    /// Failure reason
    pub failure_reason: FailureReason,
    /// Environmental conditions during failure
    pub conditions: Vec<f32>,
    /// Failure timestamp
    pub timestamp: u64,
    /// Lessons learned
    pub lessons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureReason {
    /// Insufficient information
    InsufficientInformation,
    /// Conflicting constraints
    ConflictingConstraints,
    /// Resource limitations
    ResourceLimitations,
    /// Unexpected side effects
    UnexpectedSideEffects,
    /// Environmental instability
    EnvironmentalInstability,
}

/// Adaptation analysis result
#[derive(Debug, Clone)]
pub struct AdaptationAnalysis {
    pub structural_requirements: StructuralNeeds,
    pub relational_requirements: RelationalNeeds,
    pub urgency: f32,
    pub complexity: f32,
    pub environmental_context: EnvironmentState<crate::Backend>,
}

/// Structural adaptation needs
#[derive(Debug, Clone)]
pub struct StructuralNeeds {
    pub modification_areas: Vec<String>,
    pub complexity_score: f32,
    pub priority_level: f32,
    pub required_resources: Vec<String>,
}

/// Relational adaptation needs
#[derive(Debug, Clone)]
pub struct RelationalNeeds {
    pub relationship_changes: Vec<String>,
    pub complexity_score: f32,
    pub interaction_patterns: Vec<String>,
    pub dependency_impacts: Vec<String>,
}

/// Overall adaptation result
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    pub success: bool,
    pub structural_changes: Vec<StructuralChange>,
    pub relational_changes: Vec<RelationalChange>,
    pub performance_impact: f32,
    pub adaptation_time: Duration,
    pub confidence: f32,
}

/// Structural adaptation result
#[derive(Debug, Clone)]
pub struct StructuralAdaptationResult {
    pub changes: Vec<StructuralChange>,
    pub success: bool,
    pub performance_impact: f32,
}

/// Relational adaptation result
#[derive(Debug, Clone)]
pub struct RelationalAdaptationResult {
    pub changes: Vec<RelationalChange>,
    pub success: bool,
    pub performance_impact: f32,
}

/// Validation result for adaptations
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub success: bool,
    pub confidence: f32,
    pub performance_impact: f32,
    pub structural_validation: ComponentValidation,
    pub relational_validation: ComponentValidation,
    pub overall_score: f32,
}

/// Component-specific validation result
#[derive(Debug, Clone)]
pub struct ComponentValidation {
    pub success: bool,
    pub confidence: f32,
    pub performance_impact: f32,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StructuralPlan {
    pub modifications: Vec<StructuralChange>,
    pub timeline: Duration,
    pub resources: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RelationalPlan {
    pub modifications: Vec<RelationalChange>,
    pub timeline: Duration,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    pub structural_plan: StructuralPlan,
    pub relational_plan: RelationalPlan,
    pub coordination_plan: CoordinationPlan,
}

#[derive(Debug, Clone)]
pub struct CoordinationPlan {
    pub execution_order: Vec<String>,
    pub synchronization_points: Vec<String>,
    pub rollback_strategy: String,
}

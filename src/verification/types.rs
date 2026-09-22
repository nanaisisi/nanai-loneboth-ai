//! Verification data structures and types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Verification result containing all check outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Overall verification success
    pub success: bool,
    /// Overall verification score
    pub overall_score: f32,
    /// Individual verification results
    pub consistency_result: Option<ConsistencyResult>,
    pub behavioral_result: Option<BehavioralResult>,
    pub structural_result: Option<StructuralResult>,
    pub performance_result: Option<PerformanceResult>,
    /// Verification timestamp
    pub timestamp: u64,
    /// Verification duration
    pub duration: Duration,
    /// Issues detected
    pub issues: Vec<VerificationIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Consistency verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    /// Consistency score
    pub score: f32,
    /// Violations detected
    pub violations: Vec<ConsistencyViolation>,
    /// Consistency trends
    pub trends: Vec<f32>,
    /// Corrective actions suggested
    pub corrections: Vec<String>,
}

/// Behavioral validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralResult {
    /// Behavioral coherence score
    pub coherence_score: f32,
    /// Action validity score
    pub action_validity: f32,
    /// Decision consistency score
    pub decision_consistency: f32,
    /// Behavioral anomalies
    pub anomalies: Vec<BehavioralAnomaly>,
}

/// Structural integrity result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralResult {
    /// Structural integrity score
    pub integrity_score: f32,
    /// Component health scores
    pub component_health: HashMap<String, f32>,
    /// Relationship health scores
    pub relationship_health: HashMap<String, f32>,
    /// Structural issues
    pub issues: Vec<StructuralIssue>,
}

/// Performance verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceResult {
    /// Overall performance score
    pub performance_score: f32,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f32>,
}

/// Verification issue detected during checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationIssue {
    /// Issue identifier
    pub id: String,
    /// Issue type
    pub issue_type: IssueType,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Suggested resolution
    pub resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    Consistency,
    Behavioral,
    Structural,
    Performance,
    Security,
    Resource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Consistency violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Violation magnitude
    pub magnitude: f32,
    /// Components involved
    pub components: Vec<String>,
    /// Violation context
    pub context: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ViolationType {
    StateInconsistency,
    ActionInconsistency,
    TemporalInconsistency,
    LogicalInconsistency,
    DataInconsistency,
}

/// Behavioral anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Anomaly score
    pub score: f32,
    /// Anomaly description
    pub description: String,
    /// Detection timestamp
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UnexpectedBehavior,
    DeviationFromPattern,
    PerformanceDegradation,
    ResourceAnomalY,
    CoordinationFailure,
}

/// Structural issue in system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralIssue {
    /// Component name
    pub component: String,
    /// Issue type
    pub issue_type: StructuralIssueType,
    /// Issue severity
    pub severity: f32,
    /// Issue description
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuralIssueType {
    ComponentFailure,
    RelationshipBreakdown,
    IntegrityLoss,
    ConfigurationError,
    ResourceExhaustion,
}

/// Performance bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck location
    pub location: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Impact severity
    pub impact: f32,
    /// Suggested optimization
    pub optimization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    ComputationalBottleneck,
    MemoryBottleneck,
    IOBottleneck,
    NetworkBottleneck,
    AlgorithmicBottleneck,
}

/// Efficiency metrics for performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Computational efficiency
    pub computational_efficiency: f32,
    /// Memory efficiency
    pub memory_efficiency: f32,
    /// Time efficiency
    pub time_efficiency: f32,
    /// Resource efficiency
    pub resource_efficiency: f32,
    /// Overall efficiency
    pub overall_efficiency: f32,
}

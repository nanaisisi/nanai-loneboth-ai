//! Action space, action categories and action constraints

use serde::{Deserialize, Serialize};

/// Action space definition and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpace {
    /// Dimension of action space
    pub dimension: usize,
    /// Action bounds (min, max) for each dimension
    pub bounds: Vec<(f32, f32)>,
    /// Action categories
    pub categories: Vec<ActionCategory>,
    /// Discrete vs continuous action types
    pub action_types: Vec<ActionType>,
}

/// Action category definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCategory {
    pub name: String,
    pub description: String,
    pub priority: f32,
    pub constraints: Vec<ActionConstraint>,
}

/// Action type specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Continuous { min: f32, max: f32 },
    Discrete { options: Vec<String> },
    Binary,
}

/// Action constraints for safe execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConstraint {
    pub constraint_type: ConstraintType,
    pub parameters: Vec<f32>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    Range,
    Magnitude,
    Rate,
    Dependency,
}

impl ActionSpace {
    pub fn new(dimension: usize) -> Self {
        let bounds = vec![(-1.0, 1.0); dimension];
        let action_types = vec![
            ActionType::Continuous {
                min: -1.0,
                max: 1.0,
            };
            dimension
        ];

        Self {
            dimension,
            bounds,
            categories: Vec::new(),
            action_types,
        }
    }

    /// Check if action is within bounds
    pub fn is_valid_action(&self, action: &[f32]) -> bool {
        if action.len() != self.dimension {
            return false;
        }

        action
            .iter()
            .zip(&self.bounds)
            .all(|(a, (min, max))| a >= min && a <= max)
    }

    /// Clip action to valid bounds
    pub fn clip_action(&self, action: &mut [f32]) {
        for (a, (min, max)) in action.iter_mut().zip(&self.bounds) {
            *a = a.clamp(*min, *max);
        }
    }
}

impl Default for ActionSpace {
    fn default() -> Self {
        Self::new(1)
    }
}

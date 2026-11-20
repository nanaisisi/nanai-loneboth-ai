//! Behavior module defining action patterns and policy networks
//!
//! Implements behavioral patterns, action spaces, and policy networks using burn
//! with focus on structural relationships and environmental adaptation.

pub mod action_space;
pub mod patterns;
pub mod policy;

pub use action_space::{ActionCategory, ActionConstraint, ActionSpace, ActionType, ConstraintType};
pub use patterns::{
    ActionGenerator, BehaviorPattern, BehaviorType, BehavioralMemory, PatternRecognizer,
};
pub use policy::{PolicyHead, PolicyNetwork, StateEncoder, ValueHead};

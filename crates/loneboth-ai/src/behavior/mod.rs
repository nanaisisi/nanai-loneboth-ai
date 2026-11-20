//! Behavior module defining action patterns and policy networks
//!
//! Implements behavioral patterns, action spaces, and policy networks using burn
//! with focus on structural relationships and environmental adaptation.

pub mod action_space;
pub mod policy;
pub mod patterns;

pub use action_space::{ActionCategory, ActionConstraint, ActionSpace, ActionType, ConstraintType};
pub use policy::{PolicyHead, PolicyNetwork, StateEncoder, ValueHead};
pub use patterns::{
    ActionGenerator, BehaviorPattern, BehaviorType, BehavioralMemory, PatternRecognizer,
};

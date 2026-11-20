//! Structural integrity verification for system components

use crate::{LonebothResult, SystemConfig};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};

/// Structural integrity verification for system components
#[derive(Module, Debug)]
pub struct StructuralIntegrityChecker<B: Backend> {
    /// Component analysis network
    component_analyzer: burn::nn::Linear<B>,
    /// Relationship validation network
    relationship_validator: burn::nn::Linear<B>,
    /// Integrity scoring network
    integrity_scorer: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

impl<B: Backend> StructuralIntegrityChecker<B> {
    pub(crate) fn new(config: &SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
        let obs_dim = config.environment.observation_dimension;
        let hidden_dim = obs_dim / 2;

        Ok(Self {
            component_analyzer: burn::nn::LinearConfig::new(obs_dim, hidden_dim).init(device),
            relationship_validator: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            integrity_scorer: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            activation: burn::nn::Relu::new(),
        })
    }

    pub(crate) async fn check_integrity(&self, observation: &Tensor<B, 2>) -> LonebothResult<f32> {
        let components = self.activation.forward(self.component_analyzer.forward(observation.clone()));
        let relationships = self.activation.forward(self.relationship_validator.forward(components));
        let integrity_score = self.integrity_scorer.forward(relationships).sigmoid().into_scalar();

        Ok(integrity_score)
    }
}

//! Behavioral validation for action and decision consistency

use crate::{LonebothResult, SystemConfig};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};

/// Behavioral validation for action and decision consistency
#[derive(Module, Debug)]
pub struct BehavioralValidator<B: Backend> {
    /// Behavioral pattern analyzer
    pattern_analyzer: burn::nn::Linear<B>,
    /// Action validation network
    action_validator: burn::nn::Linear<B>,
    /// Decision coherence checker
    coherence_checker: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

impl<B: Backend> BehavioralValidator<B> {
    pub(crate) fn new(config: &SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
        let obs_dim = config.environment.observation_dimension;
        let hidden_dim = obs_dim / 2;

        Ok(Self {
            pattern_analyzer: burn::nn::LinearConfig::new(obs_dim, hidden_dim).init(device),
            action_validator: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            coherence_checker: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            activation: burn::nn::Relu::new(),
        })
    }

    pub(crate) async fn validate(&self, observation: &Tensor<B, 2>) -> LonebothResult<(f32, f32, f32)> {
        let patterns = self.activation.forward(self.pattern_analyzer.forward(observation.clone()));

        let action_validity = self.action_validator.forward(patterns.clone()).sigmoid().into_scalar();
        let coherence_score = self.coherence_checker.forward(patterns).sigmoid().into_scalar();
        let decision_consistency = (action_validity + coherence_score) / 2.0;

        Ok((coherence_score, action_validity, decision_consistency))
    }
}

//! Consistency verification neural network

use crate::{LonebothResult, SystemConfig};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};

/// Neural network for consistency verification
#[derive(Module, Debug)]
pub struct ConsistencyVerifier<B: Backend> {
    /// Consistency analysis network
    analysis_network: burn::nn::Linear<B>,
    /// Violation detection network
    violation_detector: burn::nn::Linear<B>,
    /// Consistency scoring network
    scoring_network: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

impl<B: Backend> ConsistencyVerifier<B> {
    pub(crate) fn new(config: &SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
        let obs_dim = config.environment.observation_dimension;
        let hidden_dim = obs_dim / 2;

        Ok(Self {
            analysis_network: burn::nn::LinearConfig::new(obs_dim, hidden_dim).init(device),
            violation_detector: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            scoring_network: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            activation: burn::nn::Relu::new(),
        })
    }

    pub(crate) async fn verify(&self, observation: &Tensor<B, 2>) -> LonebothResult<f32> {
        let analyzed = self.activation.forward(self.analysis_network.forward(observation.clone()));
        let violations = self.activation.forward(self.violation_detector.forward(analyzed));
        let score = self.scoring_network.forward(violations);

        // Apply sigmoid to get score between 0 and 1
        let consistency_score = score.sigmoid().into_scalar();
        Ok(consistency_score)
    }
}

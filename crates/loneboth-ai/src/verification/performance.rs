//! Performance verification for system efficiency and effectiveness

use crate::{LonebothResult, SystemConfig};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};

/// Performance verification for system efficiency and effectiveness
#[derive(Module, Debug)]
pub struct PerformanceVerifier<B: Backend> {
    /// Performance analysis network
    performance_analyzer: burn::nn::Linear<B>,
    /// Efficiency checker
    efficiency_checker: burn::nn::Linear<B>,
    /// Bottleneck detector
    bottleneck_detector: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

impl<B: Backend> PerformanceVerifier<B> {
    pub(crate) fn new(config: &SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
        let obs_dim = config.environment.observation_dimension;
        let hidden_dim = obs_dim / 2;

        Ok(Self {
            performance_analyzer: burn::nn::LinearConfig::new(obs_dim, hidden_dim).init(device),
            efficiency_checker: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            bottleneck_detector: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            activation: burn::nn::Relu::new(),
        })
    }

    pub(crate) async fn verify_performance(&self, observation: &Tensor<B, 2>) -> LonebothResult<f32> {
        let performance = self.activation.forward(self.performance_analyzer.forward(observation.clone()));
        let efficiency = self.activation.forward(self.efficiency_checker.forward(performance));
        let performance_score = self.bottleneck_detector.forward(efficiency).sigmoid().into_scalar();

        Ok(performance_score)
    }
}

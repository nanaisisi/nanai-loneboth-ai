//! Policy network for action selection and value estimation

use crate::{LonebothResult, SystemConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};
use tracing::{debug, info};

/// Policy network for action selection and decision making
#[derive(Module, Debug)]
pub struct PolicyNetwork {
    /// State encoding network
    pub(crate) state_encoder: StateEncoder,
    /// Policy head for action probabilities
    pub(crate) policy_head: PolicyHead,
    /// Value estimation head
    pub(crate) value_head: ValueHead,
    /// Dropout for regularization
    pub(crate) dropout: Dropout,
    /// Configuration
    pub(crate) config: SystemConfig,
}

#[derive(Module, Debug)]
pub struct StateEncoder {
    /// Input processing
    input_layer: Linear,
    /// Hidden encoding layers
    encoding_layers: Vec<Linear>,
    /// Output encoding
    output_layer: Linear,
    activation: Relu,
}

#[derive(Module, Debug)]
pub struct PolicyHead {
    /// Policy layers
    policy_layers: Vec<Linear>,
    /// Action distribution parameters
    action_distribution: Linear,
    activation: Relu,
}

#[derive(Module, Debug)]
pub struct ValueHead {
    /// Value estimation layers
    value_layers: Vec<Linear>,
    /// Value output
    value_output: Linear,
    activation: Relu,
}

impl PolicyNetwork {
    pub fn new(config: &SystemConfig, device: &Device) -> LonebothResult<Self> {
        info!("Initializing PolicyNetwork");

        let state_dim = config.environment.observation_dimension;
        let action_dim = config.environment.action_dimension;
        let hidden_dim = (state_dim + action_dim) / 2;

        Ok(Self {
            state_encoder: StateEncoder::new(state_dim, hidden_dim, device),
            policy_head: PolicyHead::new(hidden_dim, action_dim, device),
            value_head: ValueHead::new(hidden_dim, 1, device),
            dropout: DropoutConfig::new(0.1).init(),
            config: config.clone(),
        })
    }

    /// Forward pass through policy network
    pub fn forward(&self, state: Tensor<2>) -> LonebothResult<(Tensor<2>, Tensor<2>)> {
        debug!("PolicyNetwork forward pass");

        // Encode state
        let encoded_state = self.state_encoder.forward(state);
        let encoded_state = self.dropout.forward(encoded_state);

        // Generate policy and value predictions
        let action_probs = self.policy_head.forward(encoded_state.clone());
        let state_value = self.value_head.forward(encoded_state);

        Ok((action_probs, state_value))
    }

    /// Sample action from policy
    pub fn sample_action(&self, state: Tensor<2>) -> LonebothResult<Tensor<2>> {
        let (action_probs, _) = self.forward(state)?;
        Ok(action_probs)
    }
}

impl StateEncoder {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device) -> Self {
        Self {
            input_layer: LinearConfig::new(input_dim, hidden_dim).init(device),
            encoding_layers: vec![
                LinearConfig::new(hidden_dim, hidden_dim).init(device),
                LinearConfig::new(hidden_dim, hidden_dim).init(device),
            ],
            output_layer: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            activation: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let mut x = self.activation.forward(self.input_layer.forward(input));

        for layer in &self.encoding_layers {
            x = self.activation.forward(layer.forward(x));
        }

        self.output_layer.forward(x)
    }
}

impl PolicyHead {
    fn new(input_dim: usize, action_dim: usize, device: &Device) -> Self {
        let hidden_dim = (input_dim + action_dim) / 2;

        Self {
            policy_layers: vec![
                LinearConfig::new(input_dim, hidden_dim).init(device),
                LinearConfig::new(hidden_dim, hidden_dim).init(device),
            ],
            action_distribution: LinearConfig::new(hidden_dim, action_dim).init(device),
            activation: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let mut x = input;

        for layer in &self.policy_layers {
            x = self.activation.forward(layer.forward(x));
        }

        // Apply softmax for action probabilities
        burn::tensor::activation::softmax(self.action_distribution.forward(x), 1)
    }
}

impl ValueHead {
    fn new(input_dim: usize, output_dim: usize, device: &Device) -> Self {
        let hidden_dim = input_dim / 2;

        Self {
            value_layers: vec![
                LinearConfig::new(input_dim, hidden_dim).init(device),
                LinearConfig::new(hidden_dim, hidden_dim).init(device),
            ],
            value_output: LinearConfig::new(hidden_dim, output_dim).init(device),
            activation: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let mut x = input;

        for layer in &self.value_layers {
            x = self.activation.forward(layer.forward(x));
        }

        self.value_output.forward(x)
    }
}

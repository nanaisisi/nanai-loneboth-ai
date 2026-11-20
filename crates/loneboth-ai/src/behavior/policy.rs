//! Policy network for action selection and value estimation

use crate::{LonebothResult, SystemConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::tensor::{Device, Tensor};
use tracing::{debug, info};

/// Policy network for action selection and decision making
#[derive(Module, Debug)]
pub struct PolicyNetwork<B: Backend> {
    /// State encoding network
    pub(crate) state_encoder: StateEncoder<B>,
    /// Policy head for action probabilities
    pub(crate) policy_head: PolicyHead<B>,
    /// Value estimation head
    pub(crate) value_head: ValueHead<B>,
    /// Dropout for regularization
    pub(crate) dropout: Dropout,
    /// Configuration
    pub(crate) config: SystemConfig,
}

#[derive(Module, Debug)]
pub struct StateEncoder<B: Backend> {
    /// Input processing
    input_layer: Linear<B>,
    /// Hidden encoding layers
    encoding_layers: Vec<Linear<B>>,
    /// Output encoding
    output_layer: Linear<B>,
    activation: Relu,
}

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    /// Policy layers
    policy_layers: Vec<Linear<B>>,
    /// Action distribution parameters
    action_distribution: Linear<B>,
    activation: Relu,
}

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
    /// Value estimation layers
    value_layers: Vec<Linear<B>>,
    /// Value output
    value_output: Linear<B>,
    activation: Relu,
}

impl<B: Backend> PolicyNetwork<B> {
    pub fn new(config: &SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
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
    pub fn forward(&self, state: Tensor<B, 2>) -> LonebothResult<(Tensor<B, 2>, Tensor<B, 2>)> {
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
    pub fn sample_action(&self, state: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        let (action_probs, _) = self.forward(state)?;
        Ok(action_probs)
    }
}

impl<B: Backend> StateEncoder<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device<B>) -> Self {
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

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.activation.forward(self.input_layer.forward(input));

        for layer in &self.encoding_layers {
            x = self.activation.forward(layer.forward(x));
        }

        self.output_layer.forward(x)
    }
}

impl<B: Backend> PolicyHead<B> {
    fn new(input_dim: usize, action_dim: usize, device: &Device<B>) -> Self {
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

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for layer in &self.policy_layers {
            x = self.activation.forward(layer.forward(x));
        }

        // Apply softmax for action probabilities
        self.action_distribution.forward(x).softmax(1)
    }
}

impl<B: Backend> ValueHead<B> {
    fn new(input_dim: usize, output_dim: usize, device: &Device<B>) -> Self {
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

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for layer in &self.value_layers {
            x = self.activation.forward(layer.forward(x));
        }

        self.value_output.forward(x)
    }
}

//! Decision optimization, action refinement, and uncertainty estimation

use crate::LonebothResult;
use burn::prelude::*;
use burn::tensor::Device;
use tracing::debug;

/// Decision optimization and action refinement system
#[derive(Module, Debug)]
pub struct DecisionOptimizer<B: Backend> {
    /// Action refinement network
    action_refiner: ActionRefiner<B>,
    /// Uncertainty estimation
    uncertainty_estimator: UncertaintyEstimator<B>,
    /// Multi-step planning network
    planner: MultiStepPlanner<B>,
    /// Real-time adaptation controller
    adaptation_controller: RealtimeAdaptationController<B>,
}

#[derive(Module, Debug)]
pub struct ActionRefiner<B: Backend> {
    /// Primary refinement network
    refinement_network: burn::nn::Linear<B>,
    /// Constraint enforcement layer
    constraint_layer: burn::nn::Linear<B>,
    /// Optimization layer
    optimization_layer: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

#[derive(Module, Debug)]
pub struct UncertaintyEstimator<B: Backend> {
    /// Uncertainty quantification network
    uncertainty_network: burn::nn::Linear<B>,
    /// Confidence estimation
    confidence_estimator: burn::nn::Linear<B>,
    /// Epistemic uncertainty estimation
    epistemic_estimator: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

#[derive(Module, Debug)]
pub struct MultiStepPlanner<B: Backend> {
    /// Future state predictor
    state_predictor: burn::nn::Linear<B>,
    /// Multi-horizon planning network
    planning_network: burn::nn::Linear<B>,
    /// Plan optimization
    plan_optimizer: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

#[derive(Module, Debug)]
pub struct RealtimeAdaptationController<B: Backend> {
    /// Adaptation trigger detector
    trigger_detector: burn::nn::Linear<B>,
    /// Real-time adaptation network
    adaptation_network: burn::nn::Linear<B>,
    /// Adaptation control signals
    control_signals: burn::nn::Linear<B>,
    activation: burn::nn::Relu,
}

impl<B: Backend> DecisionOptimizer<B> {
    pub(crate) fn new(config: &crate::SystemConfig, device: &Device<B>) -> LonebothResult<Self> {
        let obs_dim = config.environment.observation_dimension;
        let action_dim = config.environment.action_dimension;
        let hidden_dim = (obs_dim + action_dim) / 2;

        Ok(Self {
            action_refiner: ActionRefiner::new(action_dim, hidden_dim, device),
            uncertainty_estimator: UncertaintyEstimator::new(action_dim, hidden_dim, device),
            planner: MultiStepPlanner::new(obs_dim, hidden_dim, device),
            adaptation_controller: RealtimeAdaptationController::new(obs_dim, hidden_dim, device),
        })
    }

    /// Optimize decision based on policy output and behavioral context
    pub(crate) async fn optimize_decision(
        &self,
        action_probs: Tensor<B, 2>,
        _state_value: Tensor<B, 2>,
        behavior_result: &(crate::behavior::BehaviorType, f32),
    ) -> LonebothResult<Tensor<B, 2>> {
        debug!("Optimizing decision with behavior: {:?}", behavior_result.0);

        // Refine action based on behavioral context
        let refined_action = self
            .action_refiner
            .refine(action_probs, behavior_result.1)
            .await?;

        // Estimate uncertainty
        let uncertainty = self
            .uncertainty_estimator
            .estimate(refined_action.clone())
            .await?;

        // Apply uncertainty-based adjustments
        let final_action = if uncertainty.mean().into_scalar() > 0.5 {
            self.apply_conservative_adjustment(refined_action).await?
        } else {
            refined_action
        };

        Ok(final_action)
    }

    /// Apply real-time adaptation
    pub(crate) async fn apply_realtime_adaptation(
        &self,
        action: Tensor<B, 2>,
        observations: Tensor<B, 2>,
    ) -> LonebothResult<Tensor<B, 2>> {
        debug!("Applying real-time adaptation");

        // Detect need for adaptation
        let adaptation_signal = self
            .adaptation_controller
            .detect_adaptation_need(observations)
            .await?;

        if adaptation_signal.mean().into_scalar() > 0.3 {
            // Apply adaptation to action
            let adapted_action = self
                .adaptation_controller
                .adapt_action(action, adaptation_signal)
                .await?;
            Ok(adapted_action)
        } else {
            Ok(action)
        }
    }

    /// Apply conservative adjustment for high uncertainty
    async fn apply_conservative_adjustment(
        &self,
        action: Tensor<B, 2>,
    ) -> LonebothResult<Tensor<B, 2>> {
        // Scale down action magnitude for conservative behavior
        let conservative_action = action * 0.8;
        Ok(conservative_action)
    }
}

impl<B: Backend> ActionRefiner<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device<B>) -> Self {
        Self {
            refinement_network: burn::nn::LinearConfig::new(input_dim, hidden_dim).init(device),
            constraint_layer: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            optimization_layer: burn::nn::LinearConfig::new(hidden_dim, input_dim).init(device),
            activation: burn::nn::Relu::new(),
        }
    }

    async fn refine(&self, action: Tensor<B, 2>, confidence: f32) -> LonebothResult<Tensor<B, 2>> {
        let mut refined = self
            .activation
            .forward(self.refinement_network.forward(action));
        refined = self
            .activation
            .forward(self.constraint_layer.forward(refined));
        let final_action = self.optimization_layer.forward(refined);

        // Apply confidence-based scaling
        let confidence_tensor = Tensor::full([1, 1], confidence, &action.device());
        let scaled_action = final_action * confidence_tensor;

        Ok(scaled_action)
    }
}

impl<B: Backend> UncertaintyEstimator<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device<B>) -> Self {
        Self {
            uncertainty_network: burn::nn::LinearConfig::new(input_dim, hidden_dim).init(device),
            confidence_estimator: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            epistemic_estimator: burn::nn::LinearConfig::new(hidden_dim, 1).init(device),
            activation: burn::nn::Relu::new(),
        }
    }

    async fn estimate(&self, action: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>> {
        let features = self
            .activation
            .forward(self.uncertainty_network.forward(action));
        let uncertainty = self.confidence_estimator.forward(features);
        Ok(uncertainty)
    }
}

impl<B: Backend> MultiStepPlanner<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device<B>) -> Self {
        Self {
            state_predictor: burn::nn::LinearConfig::new(input_dim, hidden_dim).init(device),
            planning_network: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            plan_optimizer: burn::nn::LinearConfig::new(hidden_dim, input_dim).init(device),
            activation: burn::nn::Relu::new(),
        }
    }
}

impl<B: Backend> RealtimeAdaptationController<B> {
    fn new(input_dim: usize, hidden_dim: usize, device: &Device<B>) -> Self {
        Self {
            trigger_detector: burn::nn::LinearConfig::new(input_dim, hidden_dim).init(device),
            adaptation_network: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            control_signals: burn::nn::LinearConfig::new(hidden_dim, input_dim).init(device),
            activation: burn::nn::Relu::new(),
        }
    }

    async fn detect_adaptation_need(
        &self,
        observations: Tensor<B, 2>,
    ) -> LonebothResult<Tensor<B, 2>> {
        let features = self
            .activation
            .forward(self.trigger_detector.forward(observations));
        let adaptation_signal = self.control_signals.forward(features);
        Ok(adaptation_signal)
    }

    async fn adapt_action(
        &self,
        action: Tensor<B, 2>,
        adaptation_signal: Tensor<B, 2>,
    ) -> LonebothResult<Tensor<B, 2>> {
        let adaptation_features = self
            .activation
            .forward(self.adaptation_network.forward(adaptation_signal));
        let adaptation_adjustment = self.control_signals.forward(adaptation_features);
        let adapted_action = action + adaptation_adjustment * 0.1; // Small adaptation step
        Ok(adapted_action)
    }
}

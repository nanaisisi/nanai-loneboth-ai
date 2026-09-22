# API Documentation

## Overview

The Loneboth AI framework provides an action-oriented neural coordination and adaptation framework built on [Burn](https://burn.dev). It exposes high-level abstractions for environmental state tracking, structural-relational action synthesis, real-time adaptation, multi-agent coordination, and policy learning.

---

## Central Orchestrator: `LonebothAI<B: Backend>`

The main entry point for the framework (`src/lib.rs`).

```rust
use loneboth_ai::{LonebothAI, SystemConfig, Backend};
use burn::tensor::Device;

// Initialize with default configuration
let device = Device::<Backend>::default();
let ai = LonebothAI::<Backend>::new(device)?;

// Initialize with custom configuration
let config = SystemConfig::default();
let ai = LonebothAI::<Backend>::with_config(config, device)?;
```

### Core Execution Methods

#### `execute_action`

Executes an action in the environment given an observation tensor, automatically triggers context tracking, and initiates adaptation if environmental shifts exceed threshold.

```rust
pub async fn execute_action(&mut self, observation: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>>
```

#### `train`

Trains policy networks and behavioral patterns using collected state-action experience pairs.

```rust
pub async fn train(&mut self, experience_data: Vec<(Tensor<B, 2>, Tensor<B, 2>)>) -> LonebothResult<()>
```

#### `infer`

Performs low-latency real-time inference on observation tensors without triggering training pipelines.

```rust
pub async fn infer(&self, observations: Tensor<B, 2>) -> LonebothResult<Tensor<B, 2>>
```

#### `get_metrics`

Returns real-time execution statistics.

```rust
pub fn get_metrics(&self) -> SystemMetrics
```

---

## Configuration Architecture

### `SystemConfig`

Hierarchical configuration defining operational parameters:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub device: DeviceConfig,
    pub environment: EnvironmentConfig,
    pub adaptation: AdaptationConfig,
    pub training: TrainingConfig,
    pub inference: InferenceConfig,
}
```

### Sub-Configurations

- **`DeviceConfig`**:
  - `use_gpu: bool`
  - `backend_preference: BackendType` (`Candle`, `Wgpu`, `Auto`)
- **`EnvironmentConfig`**:
  - `observation_dimension: usize` (default: 128)
  - `action_dimension: usize` (default: 32)
  - `temporal_context_length: usize` (default: 10)
  - `adaptation_threshold: f32` (default: 0.1)
- **`AdaptationConfig`**:
  - `learning_rate: f64` (default: 1e-3)
  - `adaptation_frequency: usize` (default: 100)
  - `structural_weight: f32` (default: 0.6)
  - `relational_weight: f32` (default: 0.4)
- **`TrainingConfig`**:
  - `batch_size: usize` (default: 32)
  - `epochs: usize` (default: 100)
  - `validation_split: f32` (default: 0.2)
  - `early_stopping: bool` (default: true)
- **`InferenceConfig`**:
  - `batch_optimization: bool`
  - `cache_policy_states: bool`
  - `real_time_adaptation: bool`

---

## Subsystem APIs

### 1. Coordination System (`src/coordination.rs`)

Manages consensus and multi-agent collective behavior.

```rust
use loneboth_ai::coordination::{CoordinationSystem, CoordinationMode};

let coord = CoordinationSystem::new(CoordinationMode::Group, &config, device)?;
let current_mode = coord.mode();
```

**Coordination Modes:**

- `Individual`: Independent single-agent execution.
- `Group`: Distributed consensus across agents.
- `Hybrid`: Autonomous execution with periodic group synchronization.
- `Hierarchical`: Tiered decision structure.
- `Emergent`: Self-organizing behavioral convergence.
- `Adaptive`: Context-aware dynamic mode selection.

### 2. Verification System (`src/verification.rs`)

Ensures system consistency, behavioral safety, and structural integrity.

```rust
use loneboth_ai::verification::{VerificationSystem, VerificationConfig};

let verifier = VerificationSystem::new(verification_config, device)?;
```

### 3. Core Action Execution (`src/core.rs`)

- **`ActionExecutor<B>`**: Coordinates `StructuralProcessor`, `RelationalAnalyzer`, and `ActionSynthesizer`.
- **`EnvironmentState<B>`**: Represents current environmental state, temporal context, and detected non-stationarity magnitude.

---

## Backend Selection & Types

- **`Backend`**: Default computation backend alias (e.g. `Candle<f32>` or `Wgpu<f32, i32>`).
- **`AutodiffBackend`**: Autodiff-wrapped backend for gradient computation (`Autodiff<Backend>`).
- **`LonebothResult<T>`**: Alias for `anyhow::Result<T>`.

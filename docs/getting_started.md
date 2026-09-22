# Getting Started with Loneboth AI

This guide walks you through integrating and running the **Loneboth AI** framework in your Rust applications.

---

## 1. Prerequisites & Installation

Add `loneboth-ai` and `burn` to your `Cargo.toml`:

```toml
[dependencies]
loneboth_ai = { version = "0.1.0" }
burn = { version = "0.21.0", features = ["train", "std", "candle", "wgpu"] }
compio = { version = "0.19.2", features = ["full"] }
anyhow = "1.0.104"
```

---

## 2. Choosing a Computation Backend

Loneboth AI leverages Burn's backend abstractions. You can select either:

- **Candle** (`burn::backend::Candle`): Recommended for CUDA / Metal or CPU environments.
- **WGPU** (`burn::backend::Wgpu`): Recommended for cross-platform GPU execution (Vulkan, DirectX 12, Metal).

```rust
use loneboth_ai::Backend; // Defaults to Candle<f32>
use burn::tensor::Device;

let device = Device::<Backend>::default();
```

---

## 3. Configuring the System

You can customize dimensions, thresholds, and learning hyperparameters via `SystemConfig`:

```rust
use loneboth_ai::SystemConfig;

let mut config = SystemConfig::default();
// Observation & action dimensionality
config.environment.observation_dimension = 128;
config.environment.action_dimension = 32;

// Environmental adaptation sensitivity
config.environment.adaptation_threshold = 0.08;

// Training hyperparameters
config.training.batch_size = 64;
config.training.learning_rate = 1e-3;
```

---

## 4. End-to-End Execution Example

Here is a complete example demonstrating initialization, observation feeding, and action execution:

```rust
use loneboth_ai::{LonebothAI, SystemConfig, Backend};
use burn::tensor::{Tensor, Device};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize device & configuration
    let device = Device::<Backend>::default();
    let config = SystemConfig::default();

    // 2. Initialize Loneboth AI orchestrator
    let mut ai = LonebothAI::<Backend>::with_config(config, device.clone())?;

    // 3. Construct an observation tensor [batch_size = 1, observation_dim = 128]
    let observation = Tensor::<Backend, 2>::zeros([1, 128], &device);

    // 4. Execute action
    // Automatically updates temporal context and triggers online adaptation if threshold is exceeded
    let action = ai.execute_action(observation).await?;

    println!("Action shape: {:?}", action.shape());

    // 5. Inspect execution metrics
    let metrics = ai.get_metrics();
    println!("Total actions executed: {}", metrics.total_actions_executed);
    println!("Adaptation events triggered: {}", metrics.adaptation_events);

    Ok(())
}
```

---

## 5. Multi-Agent Coordination

For collective decision-making across agents, initialize a `CoordinationSystem`:

```rust
use loneboth_ai::coordination::{CoordinationSystem, CoordinationMode};
use loneboth_ai::SystemConfig;

let config = SystemConfig::default();
let coord = CoordinationSystem::new(CoordinationMode::Group, &config, device)?;

println!("Coordination mode active: {:?}", coord.mode());
```

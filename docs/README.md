# Loneboth AI Documentation

## Overview

Loneboth AI is an action-oriented neural framework built on [Burn](https://burn.dev), designed for individual and group coordination, dynamic environmental adaptation, and structural-relational decision pipelines with GPU acceleration (Wgpu / Candle).

## Architecture & Subsystems

1. **Action Execution Engine (`core`)** - Neural synthesis of structural and relational representations into actions.
2. **Environmental Context (`environment`)** - Temporal observation buffer and real-time environmental drift tracking.
3. **Behavioral Patterns (`behavior`)** - Policy network actor-critic heads and behavioral pattern classification.
4. **Coordination System (`coordination`)** - Multi-agent consensus, individual, hybrid, hierarchical, and emergent coordination.
5. **Continuous Adaptation (`adaptation`)** - Dynamic online model adjustment in response to environmental shifts.
6. **Safety & Verification (`verification`)** - Neural consistency checking and behavioral constraint validation.
7. **Hardware Acceleration (`gpu`)** - Multi-backend execution across Wgpu (DirectX/Vulkan) and Candle (CUDA/Metal).

## Documentation Sitemap

- [Architecture Design](architecture.md) - High-level system structure, components, and data flow.
- [API Reference](api.md) - Detailed API interfaces, configuration options, and type definitions.
- [Getting Started Guide](getting_started.md) - Setup instructions, backend configuration, and end-to-end examples.

## Basic Usage Example

```rust
use loneboth_ai::{LonebothAI, SystemConfig, Backend};
use burn::tensor::{Tensor, Device};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = Device::<Backend>::default();
    let mut ai = LonebothAI::<Backend>::new(device)?;

    // Create a 2D observation tensor [batch_size, observation_dim]
    let observation = Tensor::<Backend, 2>::zeros([1, 128], &Device::<Backend>::default());
    
    // Execute action with automatic context tracking and adaptation
    let action = ai.execute_action(observation).await?;
    println!("Executed action tensor: {:?}", action);

    Ok(())
}
```

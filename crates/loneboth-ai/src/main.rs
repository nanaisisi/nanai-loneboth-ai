use anyhow::Result;
use burn::tensor::Device;
use nanai_loneboth_ai::{LonebothAI, SystemConfig};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::default();
    let config = SystemConfig::default();
    let _ai = LonebothAI::with_config(config, device)?;

    println!("Loneboth AI initialized successfully with Burn backend");
    Ok(())
}

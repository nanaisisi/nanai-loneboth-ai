use anyhow::Result;
use burn::tensor::Device;
use nanai_loneboth_ai::{Backend, LonebothAI, SystemConfig};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::<Backend>::default();
    let config = SystemConfig::default();
    let _ai = LonebothAI::<Backend>::with_config(config, device)?;

    println!("Loneboth AI initialized successfully with Burn backend");
    Ok(())
}

//! Tests for the Loneboth AI framework

use burn::tensor::Device;
use nanai_loneboth_ai::{LonebothAI, SystemConfig};

#[test]
fn test_loneboth_ai_creation() {
    let device = Device::default();
    let ai = LonebothAI::new(device);
    assert!(ai.is_ok());
}

#[test]
fn test_loneboth_ai_with_config() {
    let device = Device::default();
    let config = SystemConfig::default();
    let ai = LonebothAI::with_config(config, device);
    assert!(ai.is_ok());
    let ai = ai.unwrap();
    assert_eq!(ai.config().environment.observation_dimension, 128);
}

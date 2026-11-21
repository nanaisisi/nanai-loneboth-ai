pub mod gna;
pub mod gpu;

use loneboth_core::native_sampler;
use loneboth_core::{HomeostasisState, SensationVector};

#[derive(Debug, Clone)]
pub struct SystemTelemetry {
    pub sensation: SensationVector,
    pub homeostasis: HomeostasisState,
    pub gna: gna::GnaHardwareStatus,
    pub gpus: Vec<gpu::GpuAdapterInfo>,
}

pub fn sample_system_telemetry() -> SystemTelemetry {
    let sensation = native_sampler::sample_current_sensation();
    let homeostasis = sensation.evaluate_homeostasis();
    let gna = gna::get_gna_status();
    let gpus = gpu::collect_gpu_adapters();

    SystemTelemetry {
        sensation,
        homeostasis,
        gna,
        gpus,
    }
}

use nanai_gna_dll_load::{GnaDevice, GnaLibrary};
use std::sync::Mutex;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GnaHardwareStatus {
    pub is_detected: bool,
    pub is_loaded: bool,
    pub device_name: String,
    pub description: String,
    pub completed_inferences: usize,
    pub avg_latency_ms: f64,
    pub throughput_ips: f64,
    pub last_test_status: String,
}

struct GnaSessionState {
    library: Option<GnaLibrary>,
    device: Option<GnaDevice>,
    completed_inferences: usize,
    avg_latency_ms: f64,
    throughput_ips: f64,
    last_test_status: String,
}

static GNA_SESSION: Mutex<GnaSessionState> = Mutex::new(GnaSessionState {
    library: None,
    device: None,
    completed_inferences: 0,
    avg_latency_ms: 0.0,
    throughput_ips: 0.0,
    last_test_status: String::new(),
});

pub fn get_gna_status() -> GnaHardwareStatus {
    let mut state = GNA_SESSION.lock().unwrap();

    let mut is_loaded = state.library.is_some();
    if !is_loaded {
        if let Ok(lib) = GnaLibrary::load_default() {
            state.library = Some(lib);
            is_loaded = true;
        }
    }

    let mut is_detected = state.device.is_some();
    if is_loaded && !is_detected {
        if let Some(ref lib) = state.library {
            if let Ok(dev) = GnaDevice::open_first(lib) {
                state.device = Some(dev);
                is_detected = true;
            }
        }
    }

    let description = if is_detected && is_loaded {
        "Intel GNA 3.0 NPU: Active & Session Held".to_string()
    } else if is_detected {
        "Intel GNA Device Detected (Session Open)".to_string()
    } else if is_loaded {
        "GNA Runtime Library Loaded (Device not found)".to_string()
    } else {
        "GNA Hardware / Driver not found".to_string()
    };

    let device_name = if is_detected {
        "Intel Gaussian & Neural Accelerator 3.0 (Device ID: 0x474E)".to_string()
    } else {
        "Intel GNA Device (not found)".to_string()
    };

    GnaHardwareStatus {
        is_detected,
        is_loaded,
        device_name,
        description,
        completed_inferences: state.completed_inferences,
        avg_latency_ms: state.avg_latency_ms,
        throughput_ips: state.throughput_ips,
        last_test_status: state.last_test_status.clone(),
    }
}

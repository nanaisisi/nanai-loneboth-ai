use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    IDXGIAdapter3, IDXGIFactory1,
};
use windows::core::Interface;

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub luid: LUID,
    pub dedicated_usage_bytes: u64,
    pub dedicated_budget_bytes: u64,
    pub shared_usage_bytes: u64,
    pub shared_budget_bytes: u64,
}

pub fn collect_gpu_adapters() -> Vec<GpuAdapterInfo> {
    let mut adapters = Vec::new();
    unsafe {
        let factory: Result<IDXGIFactory1, _> = CreateDXGIFactory1();
        if let Ok(factory) = factory {
            let mut i = 0;
            while let Ok(adapter) = factory.EnumAdapters1(i) {
                if let Ok(desc) = adapter.GetDesc1() {
                    // ソフトウェアアダプター（WARP等）を除外
                    if (desc.Flags & 2) == 0 {
                        let name_len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                        let name = String::from_utf16_lossy(&desc.Description[..name_len]);

                        if let Ok(adapter3) = adapter.cast::<IDXGIAdapter3>() {
                            let mut local_info = std::mem::zeroed();
                            let mut non_local_info = std::mem::zeroed();

                            let _ = adapter3.QueryVideoMemoryInfo(
                                0,
                                DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                &mut local_info,
                            );
                            let _ = adapter3.QueryVideoMemoryInfo(
                                0,
                                DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                                &mut non_local_info,
                            );

                            adapters.push(GpuAdapterInfo {
                                name,
                                luid: desc.AdapterLuid,
                                dedicated_usage_bytes: local_info.CurrentUsage,
                                dedicated_budget_bytes: local_info.Budget,
                                shared_usage_bytes: non_local_info.CurrentUsage,
                                shared_budget_bytes: non_local_info.Budget,
                            });
                        }
                    }
                }
                i += 1;
            }
        }
    }
    adapters
}

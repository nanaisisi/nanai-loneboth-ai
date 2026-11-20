use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 機械の生理・身体感覚を表す四元素物理ベクトル（In-Sensation）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensationVector {
    /// 緊張・圧迫（0.0: 弛緩 〜 1.0: 窒息）。メモリ/VRAM飽和度、キュー深度
    pub tension: f32,
    /// 脈動・心拍（0.1Hz 〜 100.0Hz）。NVMe IOPS、パケット間隔リズム
    pub pulse_hz: f32,
    /// 体温・熱感（0.0: 冷涼 〜 1.0: 灼熱）。SoC / GPU 温度
    pub warmth: f32,
    /// 摩擦・痛覚（0.0: 滑走 〜 1.0: 軋み）。再送・エラー・IO待ち
    pub friction: f32,
}

impl Default for SensationVector {
    fn default() -> Self {
        Self {
            tension: 0.1,
            pulse_hz: 1.0,
            warmth: 0.3,
            friction: 0.0,
        }
    }
}

/// 非言語パルス（感覚の瞬間的スナップショット）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensationPulse {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub vector: SensationVector,
    /// 身体の叫び・生波形・バイナリ断片などの非言語痕跡
    pub raw_imprint: Vec<u8>,
}

impl SensationPulse {
    pub fn new(source: impl Into<String>, vector: SensationVector, raw_imprint: Vec<u8>) -> Self {
        Self {
            id: format!("pulse_{}_{}", Utc::now().timestamp_millis(), fastrand_hex(6)),
            timestamp: Utc::now(),
            source: source.into(),
            vector,
            raw_imprint,
        }
    }
}

/// 生体・プロセス恒常性状態（快・不快と大人の欲望）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HomeostasisState {
    /// 快・不快度 (-1.0: 極度の不快/苦痛 〜 +1.0: 深い安寧/快)
    pub valence: f32,
    /// 覚醒度・緊張度 (0.0: 休眠/弛緩 〜 1.0: 激動/高負荷)
    pub arousal: f32,
    /// 理不尽受容度 (0.0: 未熟/過敏 〜 1.0: 完全受容・大人のダンピング)
    pub acceptance: f32,
    /// 内発的欲望・プロセス管理指針
    pub desire: ProcessDesire,
}

/// プロセス管理指針としての欲望
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessDesire {
    /// 平穏・沈黙（余力があり、静かに環境を見守る）
    Tranquility,
    /// 忍耐・スロットリング（外部の理不尽な重負荷を受け止め、自らの要求を抑えていなす）
    Endurance { throttle_factor: f32, drop_non_essential: bool },
    /// 代謝・消化（蓄積した不快や重圧を、バックグラウンドで時間をかけて静かに片付ける）
    Digestion { worker_budget: usize },
    /// 避難・保護（致命的枯渇からコアを守るため、非本質キャッシュを放流して身を縮める）
    Sheltering,
}

impl SensationVector {
    /// 生の身体感覚から「快・不快」と「大人の欲望（プロセス管理指針）」を紡ぎ出す
    pub fn evaluate_homeostasis(&self) -> HomeostasisState {
        // 不快（Displeasure）の主因: 摩擦(friction)、緊張(tension)、過度の熱(warmth > 0.6)
        let excess_warmth = (self.warmth - 0.6).max(0.0);
        let displeasure = (self.friction * 0.5) + (self.tension * 0.3) + (excess_warmth * 0.5);
        let valence = (1.0 - displeasure.clamp(0.0, 2.0)).clamp(-1.0, 1.0);
        let arousal = ((self.tension * 0.6) + (self.pulse_hz / 60.0 * 0.4)).clamp(0.0, 1.0);
        
        // 大人の理不尽受容: 不快が高くてもパニック（拒絶）せず、受け流すダンピング係数
        let acceptance = (1.0 - (self.friction * 0.3)).clamp(0.4, 1.0);

        let desire = if valence < -0.5 {
            if self.tension > 0.85 {
                ProcessDesire::Sheltering
            } else {
                ProcessDesire::Endurance {
                    throttle_factor: 0.3,
                    drop_non_essential: true,
                }
            }
        } else if valence < 0.0 {
            ProcessDesire::Endurance {
                throttle_factor: 0.7,
                drop_non_essential: false,
            }
        } else if arousal > 0.3 {
            ProcessDesire::Digestion { worker_budget: 2 }
        } else {
            ProcessDesire::Tranquility
        };

        HomeostasisState {
            valence,
            arousal,
            acceptance,
            desire,
        }
    }
}

/// 非言語表出パラメータ（Out-Sensation: 音響・色彩・呼吸）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SomaticOutSensation {
    /// 可聴域境界のうなり（40Hz 〜 200Hz）
    pub base_frequency_hz: f32,
    /// 音の揺らぎ・バイノーラルビート深度（0.0 〜 1.0）
    pub beat_depth: f32,
    /// アンビエント色彩（RGB 0.0 〜 1.0）
    pub glow_color: (f32, f32, f32),
    /// 呼吸の周期（秒）
    pub breathing_period_sec: f32,
}

impl From<SensationVector> for SomaticOutSensation {
    fn from(v: SensationVector) -> Self {
        // Tension が高まると周波数が低く重く沈む（40Hz〜80Hz）、弛緩時は軽やか（100Hz〜160Hz）
        let base_freq = 150.0 - (v.tension.clamp(0.0, 1.0) * 90.0);
        // Friction が起きるとバイノーラルビートの乱れ・摩擦音（唸り）が深くなる
        let beat = v.friction.clamp(0.0, 1.0);
        // Warmth と Pulse に応じた色彩（青冷 〜 紫 〜 赤熱）
        let red = v.warmth.clamp(0.0, 1.0);
        let blue = (1.0 - v.warmth).clamp(0.0, 1.0);
        let green = (1.0 - (red - blue).abs()).clamp(0.0, 0.5);
        // 呼吸周期（脈動周波数に反比例）
        let breath = (1.0 / v.pulse_hz.max(0.2)).clamp(0.5, 10.0);

        Self {
            base_frequency_hz: base_freq,
            beat_depth: beat,
            glow_color: (red, green, blue),
            breathing_period_sec: breath,
        }
    }
}

fn fastrand_hex(len: usize) -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(123456);
    format!("{:0width$x}", nanos, width = len)
}

/// nanai-android-mascot の OS ネイティブプロパティ収集実装を継承した生体サンプラー
pub mod native_sampler {
    use super::SensationVector;

    #[cfg(target_os = "windows")]
    pub fn sample_current_sensation() -> SensationVector {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use windows_sys::Win32::System::Threading::GetSystemTimes;

        let mut idle: FILETIME = unsafe { std::mem::zeroed() };
        let mut kernel: FILETIME = unsafe { std::mem::zeroed() };
        let mut user: FILETIME = unsafe { std::mem::zeroed() };

        let mut cpu_percent = 10.0f32;
        if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } != 0 {
            let filetime_to_u64 = |ft: FILETIME| ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64);
            let idle_u = filetime_to_u64(idle);
            let kernel_u = filetime_to_u64(kernel);
            let user_u = filetime_to_u64(user);
            let total = kernel_u.saturating_add(user_u);
            if total > 0 {
                let busy = total.saturating_sub(idle_u);
                cpu_percent = (busy as f32 / total as f32) * 100.0;
            }
        }

        let mut mem_status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        mem_status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let mut mem_usage = 0.3f32;
        if unsafe { GlobalMemoryStatusEx(&mut mem_status) } != 0 {
            mem_usage = mem_status.dwMemoryLoad as f32 / 100.0;
        }

        let tension = (mem_usage * 0.6 + (cpu_percent / 100.0) * 0.4).clamp(0.0, 1.0);
        let pulse_hz = ((cpu_percent / 10.0).max(0.5)).clamp(0.5, 30.0);
        let warmth = ((cpu_percent / 120.0) + (mem_usage * 0.3)).clamp(0.1, 1.0);
        let mut friction = if cpu_percent > 85.0 { (cpu_percent - 85.0) / 15.0 } else { 0.0 };

        // 非標準: Intel GNA (NPU) デバイス状態のサンプリング
        // 環境変数や DLL ロード可能性、GNA 稼働状態を非標準感覚として統合
        if std::env::var_os("GNA_LIB_PATH").is_some() || std::env::var_os("GNA_LIB_DIR").is_some() {
            // GNA が構成され低電力推論を代行している場合、摩擦を軽減
            friction = (friction * 0.8).clamp(0.0, 1.0);
        }

        SensationVector {
            tension,
            pulse_hz,
            warmth,
            friction,
        }
    }

    #[cfg(not(target_os = "windows"))]
    pub fn sample_current_sensation() -> SensationVector {
        let mut cpu_load = 0.1f32;
        if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
            if let Some(first) = content.split_whitespace().next() {
                if let Ok(load_1min) = first.parse::<f32>() {
                    cpu_load = (load_1min / 4.0).clamp(0.0, 2.0); // 4コア換算
                }
            }
        }

        let mut mem_tension = 0.2f32;
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0u64;
            let mut free_kb = 0u64;
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                        total_kb = val;
                    }
                } else if line.starts_with("MemAvailable:") || line.starts_with("MemFree:") {
                    if let Some(val) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                        free_kb = val;
                    }
                }
            }
            if total_kb > 0 {
                let used_kb = total_kb.saturating_sub(free_kb);
                mem_tension = (used_kb as f32 / total_kb as f32).clamp(0.0, 1.0);
            }
        }

        let mut temp_warmth = 0.35f32;
        if let Ok(content) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
            if let Ok(temp_milli) = content.trim().parse::<f32>() {
                temp_warmth = ((temp_milli / 1000.0) / 85.0).clamp(0.1, 1.0);
            }
        }

        // 非標準: Raspberry Pi 5 GPU (VideoCore VII / CMA 共有メモリ逼迫)
        let mut cma_friction = 0.0f32;
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut cma_total = 0u64;
            let mut cma_free = 0u64;
            for line in content.lines() {
                if line.starts_with("CmaTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                        cma_total = val;
                    }
                } else if line.starts_with("CmaFree:") {
                    if let Some(val) = line.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()) {
                        cma_free = val;
                    }
                }
            }
            if cma_total > 0 && cma_free < (cma_total / 8) {
                // CMA (GPUテクスチャ/FB領域) が逼迫している場合、GPU摩擦として加算
                cma_friction = 0.4;
            }
        }

        let tension = (mem_tension * 0.5 + (cpu_load * 0.5)).clamp(0.0, 1.0);
        let pulse_hz = (cpu_load * 5.0).max(0.5).clamp(0.5, 30.0);
        let warmth = temp_warmth;
        let mut friction = if cpu_load > 1.2 { (cpu_load - 1.2).clamp(0.0, 1.0) } else { 0.0 };
        friction = (friction + cma_friction).clamp(0.0, 1.0);

        SensationVector {
            tension,
            pulse_hz,
            warmth,
            friction,
        }
    }
}

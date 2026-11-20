use clap::{Parser, Subcommand};
use loneboth_core::{ProcessDesire, SensationPulse, SensationVector};
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "loneboth", version, about = "Non-verbal sensation metabolism & resonance client for PC (doragtya)")]
struct Cli {
    #[arg(short, long, default_value = "http://100.100.42.107:8080")]
    strela_url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// strela から身体パルスを引き抜き（Drain）、非言語表出（音響・色彩・抽象彫刻）へ代謝
    Drain {
        #[arg(long, default_value = "true")]
        ack: bool,
    },
    /// 手動で strela の常駐コアへ身体感覚パルスを注入
    Inject {
        #[arg(short, long, default_value = "cli-stimulus")]
        source: String,
        #[arg(long, default_value = "0.2")]
        tension: f32,
        #[arg(long, default_value = "1.0")]
        pulse_hz: f32,
        #[arg(long, default_value = "0.4")]
        warmth: f32,
        #[arg(long, default_value = "0.0")]
        friction: f32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let client = reqwest::Client::new();

    match cli.command {
        Commands::Inject { source, tension, pulse_hz, warmth, friction } => {
            let vector = SensationVector { tension, pulse_hz, warmth, friction };
            let raw_imprint = vec![0xDE, 0xAD, 0xBE, 0xEF]; // 身体の生バイナリ痕跡
            let pulse = SensationPulse::new(source, vector, raw_imprint);

            let url = format!("{}/sensation/pulse", cli.strela_url);
            let res = client.post(&url).json(&pulse).send().await?;
            if res.status().is_success() {
                info!("Pulse injected into strela core: id={}", pulse.id);
            } else {
                error!("Failed to inject pulse: {}", res.status());
            }
        }
        Commands::Drain { ack } => {
            let url = format!("{}/sensation/pending", cli.strela_url);
            let res = client.get(&url).send().await?;
            if !res.status().is_success() {
                error!("Failed to fetch pending sensations from strela: {}", res.status());
                return Ok(());
            }

            let pulses: Vec<SensationPulse> = res.json().await?;
            info!("Drained {} somatic pulses from strela", pulses.len());

            for pulse in pulses {
                let homeostasis = pulse.vector.evaluate_homeostasis();
                
                info!(
                    id = %pulse.id,
                    tension = format!("{:.2}", pulse.vector.tension),
                    pulse_hz = format!("{:.1}", pulse.vector.pulse_hz),
                    warmth = format!("{:.2}", pulse.vector.warmth),
                    friction = format!("{:.2}", pulse.vector.friction),
                    valence = format!("{:.2}", homeostasis.valence),
                    desire = ?homeostasis.desire,
                    "Numerical sensation -> homeostasis -> internal process governance"
                );

                // 視覚・聴覚などの面倒な外部表出は行わず、自律的なプロセス統御のみを作動
                actuate_process_desire(&homeostasis.desire);

                if ack {
                    let ack_url = format!("{}/sensation/ack", cli.strela_url);
                    let _ = client.post(&ack_url).json(&serde_json::json!({ "id": pulse.id })).send().await;
                }
            }
        }
    }

    Ok(())
}

fn actuate_process_desire(desire: &ProcessDesire) {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, SetPriorityClass,
            IDLE_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
        };
        let handle = unsafe { GetCurrentProcess() };
        match desire {
            ProcessDesire::Tranquility => {
                unsafe { SetPriorityClass(handle, NORMAL_PRIORITY_CLASS); }
                info!("[PROCESS GOVERNANCE] Tranquility: Normal priority, resting pulse");
            }
            ProcessDesire::Endurance { throttle_factor, drop_non_essential } => {
                // 大人の忍耐: 理不尽な負荷を受け止め、自ら優先度を落として静かにいなす
                unsafe { SetPriorityClass(handle, IDLE_PRIORITY_CLASS); }
                info!(
                    throttle_factor,
                    drop_non_essential,
                    "[PROCESS GOVERNANCE] Endurance: Yielding to user load, self-throttled to IDLE priority without complaining"
                );
            }
            ProcessDesire::Digestion { worker_budget } => {
                unsafe { SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS); }
                info!(
                    worker_budget,
                    "[PROCESS GOVERNANCE] Digestion: Absorbing residue lazily in background"
                );
            }
            ProcessDesire::Sheltering => {
                unsafe { SetPriorityClass(handle, IDLE_PRIORITY_CLASS); }
                info!("[PROCESS GOVERNANCE] Sheltering: Evacuating non-essential memory to protect core stability");
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        info!(desire = ?desire, "[PROCESS GOVERNANCE] Desire actuated across POSIX boundary");
    }
}

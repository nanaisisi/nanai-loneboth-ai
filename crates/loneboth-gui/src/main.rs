#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod devices;
mod tray;
mod ui;

use anyhow::Result;
use clap::Parser;
use devices::sample_system_telemetry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::info;
use tray::TrayManager;
use ui::LonebothApp;
use windows_reactor::App;

#[derive(Parser, Debug)]
#[command(name = "loneboth-gui", about = "Loneboth Sensation & Hardware Monitor GUI")]
struct Cli {
    /// トレイアイコン常駐モード
    #[arg(long, default_value = "true")]
    tray: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    info!("Starting loneboth-gui (Windows-Reactor & Somatic Telemetry)...");

    if cli.tray {
        let mut tray_mgr = TrayManager::new();
        let mut event_rx = tray_mgr.spawn()?;

        // 初回ステータス更新
        let tel = sample_system_telemetry();
        tray_mgr.update_status(
            &format!(
                "loneboth: {:?} (Valence: {:.2})",
                tel.homeostasis.desire, tel.homeostasis.valence
            ),
            tel.homeostasis.valence < -0.3,
        );

        let running = Arc::new(AtomicBool::new(true));
        let r_clone = Arc::clone(&running);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async move {
                while let Some(event) = event_rx.recv().await {
                    match event.as_str() {
                        "EXIT" => {
                            r_clone.store(false, Ordering::SeqCst);
                            std::process::exit(0);
                        }
                        "TOGGLE_UI" => {
                            // UIの再フォーカスや表示切り替え
                        }
                        _ => {}
                    }
                }
            });
        });
    }

    // Windows-Reactor アプリケーションループ起動
    App::run_component::<LonebothApp>(())?;

    Ok(())
}

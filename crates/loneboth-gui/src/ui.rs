use crate::devices::{SystemTelemetry, sample_system_telemetry};
use windows_reactor::*;

pub struct LonebothApp {
    telemetry: SystemTelemetry,
}

#[derive(Clone, Debug)]
pub enum AppMessage {
    Refresh,
}

impl Component for LonebothApp {
    type Input = ();
    type Message = AppMessage;

    fn create(_input: &Self::Input, _context: &ComponentContext<Self>) -> Self {
        Self {
            telemetry: sample_system_telemetry(),
        }
    }

    fn update(&mut self, message: Self::Message, _context: &ComponentContext<Self>) {
        match message {
            AppMessage::Refresh => {
                self.telemetry = sample_system_telemetry();
            }
        }
    }

    fn view(&self, _input: &Self::Input, context: &mut ViewContext<Self>) -> View {
        let sender = context.sender();
        let send = {
            let sender = sender.clone();
            move |msg| {
                let _ = sender.send(msg);
            }
        };

        let title = TextBlock::new()
            .text("loneboth - Non-Verbal Somatic & Hardware Monitor")
            .font_size(20.0)
            .font_weight(FontWeight::BOLD);

        let refresh_btn = Button::new()
            .on_click(move || send(AppMessage::Refresh))
            .content(TextBlock::new().text("🔄 Refresh Sensation"));

        let header = StackPanel::new()
            .orientation(Orientation::Horizontal)
            .spacing(12.0)
            .children((title, refresh_btn));

        // 1. 生体感覚ホメオスタシスパネル
        let s = &self.telemetry.sensation;
        let h = &self.telemetry.homeostasis;
        let homeostasis_title = TextBlock::new()
            .text(format!(
                "🧬 Sensation & Homeostasis [Valence: {:.2}, Arousal: {:.2}, Desire: {:?}]",
                h.valence, h.arousal, h.desire
            ))
            .font_size(15.0)
            .font_weight(FontWeight::SEMI_BOLD)
            .foreground(Color::argb(255, 60, 180, 240));

        let sensation_details = TextBlock::new()
            .text(format!(
                "Tension: {:.1}% | Pulse: {:.1} Hz | Warmth: {:.1}% | Friction: {:.1}% | Acceptance: {:.2}",
                s.tension * 100.0,
                s.pulse_hz,
                s.warmth * 100.0,
                s.friction * 100.0,
                h.acceptance
            ))
            .font_size(13.0);

        // 2. Intel GNA 3.0 NPU パネル
        let gna = &self.telemetry.gna;
        let gna_title = TextBlock::new()
            .text(format!("🧠 GNA Accelerator: {}", gna.device_name))
            .font_size(15.0)
            .font_weight(FontWeight::SEMI_BOLD)
            .foreground(if gna.is_detected {
                Color::argb(255, 70, 220, 140)
            } else {
                Color::argb(255, 230, 110, 90)
            });

        let gna_desc = TextBlock::new()
            .text(format!(
                "State: {} | Inferences: {} | Latency: {:.2} ms",
                gna.description, gna.completed_inferences, gna.avg_latency_ms
            ))
            .font_size(13.0);

        // 3. GPU (Intel Arc / DXGI) パネル
        let gpu_title = TextBlock::new()
            .text(format!("🎮 GPUs & Accelerators ({} detected)", self.telemetry.gpus.len()))
            .font_size(15.0)
            .font_weight(FontWeight::SEMI_BOLD)
            .foreground(Color::argb(255, 180, 120, 240));

        let mut gpu_rows = Vec::new();
        for (i, gpu) in self.telemetry.gpus.iter().enumerate() {
            let vram_used_mb = gpu.dedicated_usage_bytes / (1024 * 1024);
            let vram_budget_mb = gpu.dedicated_budget_bytes / (1024 * 1024);
            let shared_used_mb = gpu.shared_usage_bytes / (1024 * 1024);

            let txt = TextBlock::new()
                .text(format!(
                    "[{}] {} - Dedicated VRAM: {} / {} MB | Shared: {} MB",
                    i, gpu.name, vram_used_mb, vram_budget_mb, shared_used_mb
                ))
                .font_size(13.0);
            gpu_rows.push(txt);
        }

        let mut elements: Vec<View> = Vec::new();
        elements.push(header.into());
        elements.push(homeostasis_title.into());
        elements.push(sensation_details.into());
        elements.push(gna_title.into());
        elements.push(gna_desc.into());
        elements.push(gpu_title.into());

        for row in gpu_rows {
            elements.push(row.into());
        }

        StackPanel::new()
            .spacing(12.0)
            .margin(Thickness::uniform(16.0))
            .children(elements)
            .into()
    }
}

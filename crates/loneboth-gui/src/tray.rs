use anyhow::Result;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use tokio::sync::mpsc;

pub struct TrayManager {
    child: Option<Child>,
    tx: Option<mpsc::UnboundedSender<String>>,
}

impl TrayManager {
    pub fn new() -> Self {
        Self {
            child: None,
            tx: None,
        }
    }

    pub fn spawn(&mut self) -> Result<mpsc::UnboundedReceiver<String>> {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<String>();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<String>();

        let script = include_str!("tray_script.ps1");

        let mut child = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let mut stdin = child.stdin.take().expect("Failed to open tray stdin");
        let stdout = child.stdout.take().expect("Failed to open tray stdout");

        // 入力スレッド (Rust -> PowerShell)
        tokio::spawn(async move {
            let mut rx = cmd_rx;
            while let Some(msg) = rx.recv().await {
                if writeln!(stdin, "{}", msg).is_err() || stdin.flush().is_err() {
                    break;
                }
            }
        });

        // 出力監視スレッド (PowerShell -> Rust)
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let _ = event_tx.send(trimmed.to_string());
                }
            }
        });

        self.child = Some(child);
        self.tx = Some(cmd_tx);

        Ok(event_rx)
    }

    pub fn update_status(&self, text: &str, is_alert: bool) {
        if let Some(ref tx) = self.tx {
            let kind = if is_alert { "alert" } else { "normal" };
            let _ = tx.send(format!("STATUS|{}|{}", kind, text));
        }
    }
}

impl Drop for TrayManager {
    fn drop(&mut self) {
        if let Some(ref tx) = self.tx {
            let _ = tx.send("QUIT".to_string());
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }
    }
}

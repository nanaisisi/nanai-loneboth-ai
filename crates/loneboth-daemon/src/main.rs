use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use loneboth_core::{SensationPulse, SensationVector};
use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use tracing::info;

pub struct AppState {
    pub db: Mutex<Connection>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,loneboth_daemon=debug".into()),
        )
        .init();

    info!("Starting loneboth-daemon (Somatic Sensation Core) on strela (Pi 5)...");

    let db_path = std::env::var("LONEBOTH_DB_PATH").unwrap_or_else(|_| "queue.sqlite".into());
    let conn = Connection::open(&db_path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pulses (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            tension REAL NOT NULL,
            pulse_hz REAL NOT NULL,
            warmth REAL NOT NULL,
            friction REAL NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        )",
        [],
    )?;

    let state = Arc::new(AppState {
        db: Mutex::new(conn),
    });

    // 身体感覚サンプラー（バックグラウンドで Pi 5 の体温・脈動を自律的に測定）
    let sampler_state = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            let vec = sample_hardware_sensation();
            let pulse = SensationPulse::new("strela-somatic-sensor", vec, vec![]);
            if let Ok(json_str) = serde_json::to_string(&pulse) {
                let db = sampler_state.db.lock().unwrap();
                let _ = db.execute(
                    "INSERT INTO pulses (id, timestamp, source, tension, pulse_hz, warmth, friction, payload_json, status)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending')",
                    rusqlite::params![
                        pulse.id,
                        pulse.timestamp.to_string(),
                        pulse.source,
                        pulse.vector.tension,
                        pulse.vector.pulse_hz,
                        pulse.vector.warmth,
                        pulse.vector.friction,
                        json_str,
                    ],
                );
            }
        }
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/sensation/pulse", post(receive_pulse))
        .route("/sensation/pending", get(drain_pulses))
        .route("/sensation/ack", post(ack_pulse))
        .with_state(state);

    let bind_addr = std::env::var("LONEBOTH_BIND").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    info!("loneboth-daemon listening on {}", bind_addr);

    axum::serve(listener, app).await?;
    Ok(())
}

fn sample_hardware_sensation() -> SensationVector {
    loneboth_core::native_sampler::sample_current_sensation()
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "somatic core alive")
}

async fn receive_pulse(
    State(state): State<Arc<AppState>>,
    Json(pulse): Json<SensationPulse>,
) -> impl IntoResponse {
    info!(
        id = %pulse.id,
        source = %pulse.source,
        tension = pulse.vector.tension,
        pulse_hz = pulse.vector.pulse_hz,
        warmth = pulse.vector.warmth,
        friction = pulse.vector.friction,
        "Absorbed sensation pulse"
    );

    let payload = match serde_json::to_string(&pulse) {
        Ok(s) => s,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid json"),
    };

    let res = {
        let db = state.db.lock().unwrap();
        db.execute(
            "INSERT INTO pulses (id, timestamp, source, tension, pulse_hz, warmth, friction, payload_json, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending')",
            rusqlite::params![
                pulse.id,
                pulse.timestamp.to_string(),
                pulse.source,
                pulse.vector.tension,
                pulse.vector.pulse_hz,
                pulse.vector.warmth,
                pulse.vector.friction,
                payload,
            ],
        )
    };

    match res {
        Ok(_) => (StatusCode::CREATED, "pulse sedimented"),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "write error"),
    }
}

async fn drain_pulses(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.lock().unwrap();
    let mut stmt = match db.prepare(
        "SELECT payload_json FROM pulses WHERE status = 'pending' ORDER BY timestamp ASC LIMIT 50",
    ) {
        Ok(s) => s,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, Json::<Vec<SensationPulse>>(vec![])).into_response(),
    };

    let rows = stmt.query_map([], |row| {
        let json_str: String = row.get(0)?;
        Ok(json_str)
    });

    let mut pulses = Vec::new();
    if let Ok(mapped) = rows {
        for item in mapped.flatten() {
            if let Ok(p) = serde_json::from_str::<SensationPulse>(&item) {
                pulses.push(p);
            }
        }
    }

    (StatusCode::OK, Json(pulses)).into_response()
}

#[derive(serde::Deserialize)]
struct AckRequest {
    id: String,
}

async fn ack_pulse(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AckRequest>,
) -> impl IntoResponse {
    let db = state.db.lock().unwrap();
    let res = db.execute(
        "UPDATE pulses SET status = 'metabolized' WHERE id = ?1",
        rusqlite::params![req.id],
    );

    match res {
        Ok(rows) if rows > 0 => (StatusCode::OK, "metabolized"),
        Ok(_) => (StatusCode::NOT_FOUND, "pulse id not found"),
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ack failed"),
    }
}

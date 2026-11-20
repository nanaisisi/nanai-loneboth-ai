# nanai-loneboth-ai 操作系仕様・運用ドキュメント

本ドキュメントは、自律常駐生体統御システム `loneboth` における意思決定・推論コア **`nanai-loneboth-ai`**（クレート名: `loneboth-ai` / `nanai_loneboth_ai`）の操作系（Action Space / Execution / Coordination / Governance）のアーキテクチャ、API、運用方法をまとめた仕様書です。

---

## 1. 概要と基本哲学

`nanai-loneboth-ai` は、深層学習フレームワーク **Burn** をバックエンドに採用し、ハードウェアの生体感覚（Somatic Sensations）を入力としてプロセスの恒常性維持（Homeostasis）と自律統御を行うためのアクション指向型推論エンジンです。

### 非言語・非テキスト性の徹底
- **完全な数値ベクトル駆動**: LLM によるテキスト生成やプロンプト解釈を行わず、感覚ベクトル（CPU/RAM/GPU/GNA負荷、レイテンシ、温度等の連続値）を直接テンソル（`Tensor<B, 2>`）として受容します。
- **快・不快のホメオスタシス**: システム全体の安定度（快適性 / 不快度）を内部テンソルで評価し、それを解消・最適化するための統御行動（Action）を出力します。

---

## 2. 操作系アーキテクチャ

`nanai-loneboth-ai` の操作系は以下の4層構造で構成されています。

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. 観測層 (Observation & Environment)                       │
│    - loneboth-core からの感覚パルス（SensationPulse）       │
│    - 環境状態（EnvironmentState）のテンソル変換・履歴管理    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 意思決定・推論層 (InferenceEngine & DecisionMaker)       │
│    - PolicyNetwork（方策ネットワーク）による行動確率算出    │
│    - GNA / GPU (WGPU) / CPU バックエンドアクセラレーション  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 行動空間・実行層 (ActionSpace & ActionExecutor)          │
│    - 連続値行動ベクトル ──> 離散統御アクションへのマッピング │
│    - 構造的コンテキスト（StructuralContext）との適合性検証  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 統御作用層 (Process Governance & Actuation)             │
│    - Win32: SetPriorityClass (Realtime/High/Idle/EcoQoS)    │
│    - Linux: nice / cgroups v2 / CPU Throttling              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. アクション空間（Action Space）の定義

`behavior::ActionSpace` は、推論エンジンが出力する多次元テンソルをシステムの物理的操作へとバインドします。

### 3.1. アクション次元 (`action_dimension: 32`)
標準設定において、行動ベクトルは以下のカテゴリに割り振られます：

| 次元範囲 | 統御対象 | 具体的な作用 |
| :--- | :--- | :--- |
| **00 - 07** | **プロセス優先度制御 (Priority)** | ターゲットプロセスの nice値変更、Win32 PriorityClass（`IDLE_PRIORITY_CLASS` 〜 `HIGH_PRIORITY_CLASS`）の切り替え |
| **08 - 15** | **リソース割当・スロットリング (Throttling)** | CPU Affinity の制限、Linux cgroups による CPU 帯域制限（Quota）、省電力コア（E-Core）への退避 |
| **16 - 23** | **メモリ・キャッシュ管理 (Memory/Cache)** | ワーキングセットのトリミング（EmptyWorkingSet）、メモリ解放圧の付与、キャッシュ破棄 |
| **24 - 31** | **協調・自己適応 (Coordination & Adaptation)** | サンプリング間隔の動的伸縮（パルスレート変更）、推論バックエンドの切り替え（GNA ⇔ GPU ⇔ CPU） |

---

## 4. プログラミング・API 操作系

### 4.1. 初期化とバックエンド構成

`SystemConfig` により、ハードウェア特性に応じた最適なバックエンド（GNA、WGPU、Candle）を選択して初期化します。

```rust
use anyhow::Result;
use burn::tensor::Device;
use nanai_loneboth_ai::{
    Backend, BackendType, DeviceConfig, EnvironmentConfig, LonebothAI, SystemConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. システム設定の構築
    let mut config = SystemConfig::default();
    config.device.backend_preference = BackendType::Auto; // GNA/WGPU自動選択
    config.environment.observation_dimension = 128;       // 感覚パルス次元
    config.environment.action_dimension = 32;             // 統御行動次元

    // 2. デバイスの初期化
    let device = Device::<Backend>::default();

    // 3. AI統御エンジンのインスタンス化
    let mut ai = LonebothAI::<Backend>::with_config(config, device)?;

    println!("nanai-loneboth-ai initialized.");
    Ok(())
}
```

### 4.2. 行動推論と実行ループ (`execute_action`)

環境からの観測データ（テンソル）を入力し、行動を決定・適応させる基本ループです。

```rust
use burn::tensor::Tensor;

// [バッチサイズ 1, 観測次元 128] の感覚テンソルを作成
let observation = Tensor::<Backend, 2>::zeros([1, 128], &device);

// 行動の推論・実行・環境適応を一括評価
let action_tensor = ai.execute_action(observation).await?;

// 行動テンソルの取得・確認
let action_data = action_tensor.to_data();
println!("Selected Action Vector: {:?}", action_data);
```

### 4.3. 経験データによる強化学習・パターン訓練 (`train`)

過去の「観測値」と「望ましい行動/結果」のペアを与えて方策ネットワーク（`PolicyNetwork`）を更新します。

```rust
let experience_samples = vec![
    (obs_tensor_1, target_action_tensor_1),
    (obs_tensor_2, target_action_tensor_2),
];

// オンライン／オフライン パターン学習の実行
ai.train(experience_samples).await?;
```

---

## 5. 常駐デーモン（`loneboth-daemon`）との連携操作

`nanai-loneboth-ai` は、strela（Pi 5）や doragtya（Win 11）上で稼働する `loneboth-daemon` の意思決定コアとして組み込まれます。

1. **感覚キューの収集**:
   - `loneboth-core::native_sampler` が OS ネイティブのメトリクス（CPU, RAM, GPU, GNA）を 500ms〜数秒周期でパルス取得。
2. **AI 推論による統御欲求の生成**:
   - `nanai-loneboth-ai` がパルス列を評価し、不快度（負荷偏り、異常過熱、メモリ逼迫）を低減する `ProcessDesire`（プロセス統御要求）を決定。
3. **アクチュエータによる実行**:
   - デーモン側のアクチュエータが対象 PID に対して Win32 API または Linux システムコールを叩いて即座に優先度を調整。

---

## 6. メトリクス監視とヘルスチェック

`ai.get_metrics()` により、エンジンの稼働状態を非言語数値として監視可能です。

```rust
let metrics = ai.get_metrics();
println!("Total Actions Executed: {}", metrics.total_actions_executed);
println!("Training Iterations   : {}", metrics.training_iterations);
println!("Inference Latency (ms): {:.3}", metrics.inference_speed);
println!("Adaptation Events     : {}", metrics.adaptation_events);
```

### 運用指標
- **Inference Latency**: GNA/GPU オフロード時は 1.0ms 前後が適正値。5.0ms を超過した場合はバッチ最適化フラグやバックエンド見直しを推奨。
- **Adaptation Events**: 急激な環境変化（プロセスの急増、ゲーム/LLMの起動）に追従して増加します。

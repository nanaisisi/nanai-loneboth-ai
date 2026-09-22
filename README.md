# Loneboth AI Framework

Loneboth AI は、[Burn](https://burn.dev) ディープラーニングフレームワークを基盤とした、個体および群体（マルチエージェント）の協調、環境適応、および構造・関係性に着目した行動生成・推論・学習フレームワークです。

GPU アクセラレーション（Candle / WGPU）を標準サポートし、動的に変化する環境に対する自律的な適応（Online Adaptation）とニューラル整合性検証を提供します。

---

## 主な特徴 (Features)

- **構造・関係性の行動生成パイプライン (Structural & Relational Action Execution)**:
  - `StructuralProcessor`: 観測空間の空間・構造的特徴を抽出
  - `RelationalAnalyzer`: 時系列コンテキストと構造特徴の相関・アテンションを解析
  - `ActionSynthesizer`: 統合特徴から多次元行動テンソルを合成
- **マルチエージェント協調システム (Coordination System)**:
  - 単体 (`Individual`)、群体合意 (`Group`)、ハイブリッド (`Hybrid`)、階層型 (`Hierarchical`)、創発型 (`Emergent`)、動的適応型 (`Adaptive`)
- **環境適応・ドリフト検知 (Dynamic Environmental Adaptation)**:
  - 環境変化量（Drift Magnitude）をリアルタイムに検知し、適応閾値を超えた場合にメタ適応・オンラインファインチューニングを自動発火
- **安全性と整合性検証 (Consistency Verification)**:
  - 予測状態と現実状態の乖離度および行動制約を評価するニューラル検証ネットワーク
- **ハードウェアアクセラレーション (GPU Acceleration)**:
  - Burn の `Wgpu` (DirectX 12, Vulkan, Metal) および `Candle` (CUDA, Metal) によるシームレスな GPU 最適化

---

## クイックスタート (Quick Start)

### 依存関係の追加 (`Cargo.toml`)

```toml
[dependencies]
loneboth_ai = { version = "0.1.0" }
burn = { version = "0.21.0", features = ["train", "std", "candle", "wgpu"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

### 基本的な使い方

```rust
use loneboth_ai::{LonebothAI, SystemConfig, Backend};
use burn::tensor::{Tensor, Device};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. デバイスと設定の初期化
    let device = Device::<Backend>::default();
    let config = SystemConfig::default();

    let mut ai = LonebothAI::<Backend>::with_config(config, device.clone())?;

    // 2. 観測テンソルの作成 [batch_size: 1, obs_dim: 128]
    let observation = Tensor::<Backend, 2>::zeros([1, 128], &device);

    // 3. 行動実行（環境コンテキストの更新と適応が自動で行われます）
    let action = ai.execute_action(observation).await?;
    println!("Action Tensor: {:?}", action);

    // 4. メトリクスの取得
    let metrics = ai.get_metrics();
    println!("Actions executed: {}", metrics.total_actions_executed);

    Ok(())
}
```

---

## ドキュメント (Documentation)

- [詳細ドキュメント 目次 (docs/README.md)](docs/README.md)
- [アーキテクチャ設計書 (docs/architecture.md)](docs/architecture.md)
- [API 仕様書 (docs/api.md)](docs/api.md)
- [導入チュートリアル (docs/getting_started.md)](docs/getting_started.md)

---

## ライセンス (License)

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

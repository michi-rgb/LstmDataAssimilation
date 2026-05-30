# 日経平均株価予測 × データ同化システム

PyTorchによるLSTMモデルと、カルマンフィルタによるデータ同化を組み合わせた日経平均株価予測システムです。

## 📌 概要

- **学習期間**: 2016年1月12日 ～ 2025年10月11日（約10年間）
- **検証期間**: 2025年10月12日 ～ 2026年1月12日（93日間）
- **手法**: LSTM（PyTorch） + 誤差パターン学習付きカルマンフィルタ
- **データソース**: Yahoo Finance（CSVで1回ダウンロード）

## 🎯 実行結果

### 予測精度（MAE: 平均絶対誤差）

以下の表は代表例です。実行のたびに再計算されます。

| 手法 | MAE | RMSE | 改善率 |
|------|-----|------|--------|
| LSTM単独 | 792.01円 | 968.94円 | - |
| カルマンフィルタ予測（主指標） | 769.89円 | 934.42円 | 2.79% |
| カルマンフィルタ補正（副指標） | 154.04円 | 188.24円 | 80.55% |

### 主要な特徴

- ✅ LSTMで過去30日間のデータから次の日の株価を予測
- ✅ 予測ステップで過去のLSTM誤差系列からバイアスを学習
- ✅ 更新ステップは誤差学習と比較用途として維持
- ✅ 1日ずつデータを追加しながらシミュレーション
- ✅ 主指標（予測ステップ誤差）でLSTM対比の改善を確認

## 📁 プロジェクト構造

```
.
├── data/                          # データディレクトリ
│   ├── nikkei_raw.csv            # 日経平均生データ
│   ├── train_data.csv            # 学習用データ（正規化済み）
│   └── test_data.csv             # 検証用データ（正規化済み）
├── models/                        # モデル保存ディレクトリ
│   ├── lstm_model.pth            # 学習済みLSTMモデル
│   └── scaler.pkl                # MinMaxScaler
├── results/                       # 結果保存ディレクトリ
│   ├── training_loss.csv         # 学習曲線
│   ├── data_assimilation_results.csv  # データ同化結果
│   ├── training_loss.png              # グラフ1: LSTM学習曲線
│   ├── prediction_comparison.png      # グラフ2: 予測値 vs 実績値 vs 補正値
│   ├── prediction_error_comparison.png # グラフ3: 予測誤差の比較
│   ├── kalman_gain.png                # グラフ4: カルマンゲインの推移
│   └── kalman_error_covariance.png    # グラフ5: 推定誤差共分散の推移
├── src/                           # ソースコードディレクトリ
│   ├── download_data.py          # データダウンロード
│   ├── preprocess_data.py        # データ前処理
│   ├── train_lstm.py             # LSTM学習
│   ├── kalman_filter.py          # カルマンフィルタ実装
│   ├── data_assimilation.py      # データ同化検証
│   └── visualize_results.py      # 結果可視化
├── requirements.txt               # 依存ライブラリ
└── README.md                      # このファイル
```

## 🚀 使い方

### 1. 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

### 2. データダウンロード（初回のみ）

```bash
python src/download_data.py
```

### 3. データ前処理

```bash
python src/preprocess_data.py
```

### 4. LSTM学習

```bash
python src/train_lstm.py
```

### 5. データ同化検証

```bash
python src/data_assimilation.py
```

### 6. 結果可視化

```bash
python src/visualize_results.py
```

このコマンドで5つの独立したグラフが生成されます：
1. LSTM学習曲線
2. 予測値 vs 実績値 vs 補正値
3. 予測誤差の比較
4. カルマンゲインの推移
5. 推定誤差共分散の推移

## 📊 技術詳細

### LSTM モデル

- **入力**: 過去30日間の株価データ
- **構造**: 2層LSTM（隠れ層50ユニット）+ 全結合層
- **損失関数**: MSE（平均二乗誤差）
- **最適化器**: Adam（学習率0.001）
- **エポック数**: 50
- **バッチサイズ**: 32

### カルマンフィルタ（誤差パターン学習付き）

- **状態**: 予測値
- **観測**: 実績値
- **プロセスノイズ分散（Q）**: 0.00025
- **観測ノイズ分散（R）**: 0.00008
- **bias_gain_early / bias_gain_main**: 0.05 / 0.18
- **max_bias_correction**: 0.03

予測ステップでは、前回フィルタ状態は使わず、LSTM予測値そのものに
直近誤差パターンから推定したバイアス補正を適用します。

なお、現在の実装では予測時点の平均値算出にはカルマンフィルタの状態遷移式は使われておらず、
予測値の生成は「LSTM予測 + 誤差補正」が主体です。
カルマンフィルタとしての処理は主に更新ステップで行われ、
観測値を取り込んだ補正と誤差共分散の更新に責任を持ちます。

#### 直近誤差パターンの導出方法
1. 前日の実績値 $z_{k-1}$ と前日のLSTM予測値 $x^{LSTM}_{k-1}$ の差分を計算し、
   その誤差 $e_{k-1} = z_{k-1} - x^{LSTM}_{k-1}$ を保存します。
2. 誤差履歴は直近 `error_history_window` 日分のみを参照し、過去の誤差を蓄積します。
3. 履歴が蓄積されるほど、LSTMの系統的バイアスを安定的に捉えられるようになります。

#### バイアス補正の推定方法
- 直近の誤差に対して、古い誤差よりも新しい誤差を強く反映するために
  等間隔の重み $w_i$ を付けます。
- 重み付き平均を計算し、誤差の傾向 $b$ を推定します。
- その推定値に補正係数 $\gamma$（`bias_gain_early` または `bias_gain_main`）をかけて、
  補正量の大きさを調整します。
- 補正量が大きすぎると予測が不安定になるため、`max_bias_correction` で上下限を制限します。

数式では次のように表されます：

$$
b = \frac{\sum_{i=1}^n w_i e_{k-i}}{\sum_{i=1}^n w_i},\qquad
b_k = \operatorname{clip}(\gamma b, -c, c)
$$

ここで：
- $n$ は直近誤差のウィンドウ長
- $w_i$ は古い誤差ほど小さく、直近の誤差ほど大きい重み
- $\gamma$ は学習段階によって変わる補正係数
- $c$ は `max_bias_correction`

#### 予測ステップ
- LSTMによる当日予測 $x^{LSTM}_k$ から、推定したバイアス補正 $b_k$ を差し引きます。

$$
\hat{x}_{k|k-1} = x^{LSTM}_k - b_k
$$

更新ステップでは、観測値を使って以下の式で状態を更新します：

$$
x_{k|k} = x_{k|k-1} + K_k (z_k - x_{k|k-1})
$$

ここで：
- $x_{k|k-1}$: バイアス補正後の予測値
- $z_k$: 実績値
- $K_k$: カルマンゲイン（0～1、観測値への信頼度）
- $x_{k|k}$: 補正後の推定値

### データ正規化

- **手法**: MinMaxScaler（0～1に正規化）
- **基準**: 学習データの統計量で正規化

## 📈 実験結果の見方

### グラフ1: `results/training_loss.png`

LSTM学習の進行状況を表示します。損失が減少していることで、モデルが正しく学習されていることを確認できます。

### グラフ2: `results/prediction_comparison.png`

4つの系列を重ねて表示：
- **黒線（実績値）**: 実際の日経平均株価
- **赤線（LSTM予測値）**: LSTMが翌日に向けて生成した予測値
- **オレンジ線（カルマンフィルタ予測値）**: **実務で予測値として活用できる系列**。  
  「一期前（前日）までの実績値でフィッティングしたカルマンフィルタ」を使い、LSTM予測値を補正した値。  
  当日の実績値が判明する前に算出できるため、翌日予測として実際の運用シナリオで利用可能。
- **緑線（カルマンフィルタ補正値）**: **参考値（実務での予測には不使用）**。  
  当日の実績値も組み込んでカルマンフィルタを更新した後の後退推定値。  
  実績値が出て初めて計算できるため実務の翌日予測には使えないが、「もし実績を知っていたらどこまで補正できたか」を示す参考指標として有用。

> オレンジ線と緑線の差は「当日の実績を反映したカルマンフィルタがどれだけ追加補正できるか」を示します。この差が小さいほど、一期前フィッティングの予測精度が高いと解釈できます。

### グラフ3: `results/prediction_error_comparison.png`

LSTM予測誤差（赤棒）とカルマンフィルタ予測誤差（オレンジ棒）を主比較として表示します。
必要に応じて、カルマンフィルタ補正誤差（緑線）も副指標として重ねて表示します。

### グラフ4: `results/kalman_gain.png`

カルマンゲイン（K_k）の推移を表示します：
- 0に近い：予測値を信頼
- 1に近い：観測値（実績値）を信頼

平均値も赤点線で表示されます。

### グラフ5: `results/kalman_error_covariance.png`

推定誤差共分散（P_k）の推移を表示します。通常、観測を重ねるごとに減少し、
カルマンフィルタの信頼度が向上していきます。

## 🔧 カスタマイズ

### LSTMのハイパーパラメータ調整

[src/train_lstm.py](src/train_lstm.py) を編集：

```python
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
num_epochs = 50
batch_size = 32
seq_length = 30  # シーケンス長
```

### カルマンフィルタのパラメータ調整

[src/data_assimilation.py](src/data_assimilation.py) を編集：

```python
kf = ErrorPatternKalmanFilter(
    process_variance=0.00025,
    measurement_variance=0.00008,
    initial_value=recent_data[-1, 0],
    initial_estimate_error=0.001,
    error_history_window=10,
    bias_gain_early=0.05,
    bias_gain_main=0.18,
    max_bias_correction=0.03
)
```

※ 主指標（KF予測誤差）の改善率は学習済みLSTMモデルの状態に依存して変動します。

## 📌 注意事項

- YahooFinanceのデータは市場休業日（土日祝日）は含まれません
- 学習データとテストデータで同じスケーラー（MinMaxScaler）を使用しています
- GPUが利用可能な場合は自動的にGPUで学習します

## 📝 ライセンス

MIT License

## 👨‍💻 作成者

GitHub Copilot

# 日経平均株価予測 × 拡張状態Ensemble Kalman Filter

PyTorchによるLSTMモデルと、拡張状態Ensemble Kalman Filter（EnKF）によるデータ同化を組み合わせた日経平均株価予測システムです。

## 📌 概要

- **学習期間**: 2016年1月12日 ～ 2025年10月11日（約10年間）
- **検証期間**: 2025年10月12日 ～ 2026年1月12日（93日間）
- **手法**: LSTM（PyTorch） + 拡張状態Ensemble Kalman Filter
- **データソース**: Yahoo Finance（CSVで1回ダウンロード）

## 🎯 実行結果

### 予測精度（MAE: 平均絶対誤差）

以下の表は代表例です。実行のたびに再計算されます。

| 手法 | MAE | RMSE | 改善率 |
|------|-----|------|--------|
| LSTM単独 | 792.01円 | 968.94円 | - |
| 拡張状態EnKF補正予測（主指標） | 750〜770円 | 920〜950円 | 3〜5% |
| 拡張状態EnKF更新値（副指標） | 150〜200円 | 180〜220円 | 75〜81% |

### 主要な特徴

- ✅ LSTMで過去30日間のデータから次の日の株価を予測
- ✅ 予測ステップで拡張状態（予測 + バイアス）から系統誤差を学習
- ✅ アンサンブル手法で複数の状態サンプルを保持し、観測から効率的に同化
- ✅ 観測前に LSTM 予測をバイアスで補正してから観測値を同化
- ✅ 1日ずつデータを追加しながらシミュレーション
- ✅ 主指標（補正予測誤差）でLSTM対比の改善を確認

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

### 拡張状態Ensemble Kalman Filter（EnKF）

#### 概要

拡張状態EnKFは、スカラー予測値とスカラー系統誤差バイアスを拡張状態として同時に推定します。
アンサンブル手法により、複数のサンプルを保持することで、観測から効率的にモデル誤差パラメータを学習します。

#### 拡張状態の定義

$$
\mathbf{x}_{k} = \begin{bmatrix} x_k \\ b_k \end{bmatrix}
$$

ここで：
- $x_k$: 予測値（LSTM予測 + バイアス補正）
- $b_k$: 系統誤差バイアス

#### 予測ステップ（観測前の補正）

LSTM予測値 $x^{\text{LSTM}}_{k}$ に対して、前回更新で学習したバイアス $b_{k-1}$ を加えて補正した予測値を生成します：

$$
x_{k|k-1} = x^{\text{LSTM}}_{k} + b_{k-1} + \nu_k^{(x)}, \quad \nu_k^{(x)} \sim \mathcal{N}(0, Q_x)
$$

バイアスプロセスはランダムウォークで進化します：

$$
b_{k|k-1} = b_{k-1} + \nu_k^{(b)}, \quad \nu_k^{(b)} \sim \mathcal{N}(0, Q_b)
$$

#### 更新ステップ（観測同化）

観測値 $z_k$ （実績株価）が得られたら、アンサンブル Kalman フィルタの更新式を適用します。

各アンサンブルメンバー $i$ に対して：

$$
\mathbf{x}_{k|k}^{(i)} = \mathbf{x}_{k|k-1}^{(i)} + \mathbf{K}_k (z_k^{(i)} - x_{k|k-1}^{(i)})
$$

ここで $\mathbf{K}_k$ は Kalman gain 行列：

$$
\mathbf{K}_k = \mathbf{P}_{xy} / (\mathbf{P}_{yy} + R)
$$

- $\mathbf{P}_{xy}$: 拡張状態と観測の共分散
- $\mathbf{P}_{yy}$: 観測の分散
- $R$: 観測ノイズ分散

#### パラメータ設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `ensemble_size` | 40 | アンサンブルメンバー数 |
| `process_variance_x` | 0.00025 | 予測値のプロセスノイズ分散 $Q_x$ |
| `process_variance_bias` | 0.000012 | バイアスのプロセスノイズ分散 $Q_b$ |
| `measurement_variance` | 0.00008 | 観測ノイズ分散 $R$ |
| `initial_state_spread` | 0.003 | 初期予測値アンサンブル拡散 |
| `initial_bias_spread` | 0.008 | 初期バイアスアンサンブル拡散 |

#### 系統誤差バイアスの導出

系統誤差バイアス $b_k$ は、LSTM予測値が観測値（実績値）に対して **持続的にずれる傾向** を定量化したパラメータです。
拡張状態EnKFでは、このバイアスを外部から手動設定するのではなく、**観測データから逐次的に推定・更新**します。

##### ステップ1：初期化

アンサンブルの初期状態を乱数で散布します：

$$
\mathbf{x}_{0}^{(i)} = \begin{bmatrix} x_0 + \epsilon_x^{(i)} \\ b_0 + \epsilon_b^{(i)} \end{bmatrix}, \quad
\epsilon_x^{(i)} \sim \mathcal{N}(0,\, \sigma_{x,0}^2), \quad
\epsilon_b^{(i)} \sim \mathcal{N}(0,\, \sigma_{b,0}^2)
$$

初期バイアス $b_0 = 0$（無バイアス）から出発し、観測を重ねるごとに収束させます。

##### ステップ2：予測ステップでのバイアス伝播

各アンサンブルメンバー $i$ に対して：

$$
x_{k|k-1}^{(i)} = x^{\text{LSTM}}_k + b_{k-1}^{(i)} + \nu_k^{(x,i)}, \quad \nu_k^{(x,i)} \sim \mathcal{N}(0,\, Q_x)
$$

$$
b_{k|k-1}^{(i)} = b_{k-1}^{(i)} + \nu_k^{(b,i)}, \quad \nu_k^{(b,i)} \sim \mathcal{N}(0,\, Q_b)
$$

バイアスはランダムウォーク（$Q_b$ の微小なノイズのみ）で進化します。これにより、バイアスは急激には変化せず、緩やかに適応します。

> **ランダムウォークとは**
>
> ランダムウォーク（Random Walk）とは、現在の値に毎ステップ独立した乱数（ノイズ）を加えていく確率過程です：
>
> $$b_k = b_{k-1} + \nu_k, \quad \nu_k \sim \mathcal{N}(0,\, Q_b)$$
>
> 平均的には前のステップの値を引き継ぎつつ（ドリフトなし）、小さなノイズで少しずつ値が変化します。
> $Q_b$ が小さいほど変化が緩やか、大きいほど変化が速くなります。
>
> **なぜバイアスにランダムウォークを使うか**
>
> - LSTM の系統誤差は**急に消えたり反転したりせず、緩やかに変化する**という仮定が自然
> - 完全に固定（$Q_b = 0$）にすると初期誤差がずっと残り、急変に追従できない
> - $Q_b > 0$ にすることで「バイアスは徐々に変わりうる」という不確かさをモデルに与え、フィルタが観測データに合わせて継続的に再学習できる
>
> 本プログラムでは `process_variance_bias` $= Q_b = 0.000012$（正規化空間）と非常に小さい値を設定しており、バイアスは1日に最大でも $\sqrt{Q_b} \approx 0.0035$ 程度しか揺れない設計になっています。

##### ステップ3：更新ステップでのバイアス学習

観測値 $z_k$（実績株価）が得られると、EnKFの更新式によってバイアスを含む拡張状態全体が修正されます。

**共分散の計算**（アンサンブル経験統計）：

$$
\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_{k|k-1}^{(i)}, \quad \bar{y} = \frac{1}{N}\sum_{i=1}^{N} x_{k|k-1}^{(i)}
$$

$$
\mathbf{P}_{xy} = \frac{1}{N-1} \sum_{i=1}^{N} \left(\mathbf{x}_{k|k-1}^{(i)} - \bar{\mathbf{x}}\right)\left(x_{k|k-1}^{(i)} - \bar{y}\right)
$$

$$
P_{yy} = \frac{1}{N-1} \sum_{i=1}^{N} \left(x_{k|k-1}^{(i)} - \bar{y}\right)^2 + R
$$

**カルマンゲイン（2次元ベクトル）**：

$$
\mathbf{K}_k = \frac{\mathbf{P}_{xy}}{P_{yy}} = \begin{bmatrix} K_k^{(x)} \\ K_k^{(b)} \end{bmatrix}
$$

- $K_k^{(x)}$：予測値 $x_k$ への補正の強さ
- $K_k^{(b)}$：バイアス $b_k$ への補正の強さ ← **ここでバイアスが学習される**

**各メンバーの更新**（観測をパータベーションで取り込む）：

$$
z_k^{(i)} = z_k + \epsilon_R^{(i)}, \quad \epsilon_R^{(i)} \sim \mathcal{N}(0,\, R)
$$

$$
\mathbf{x}_{k|k}^{(i)} = \mathbf{x}_{k|k-1}^{(i)} + \mathbf{K}_k \left(z_k^{(i)} - x_{k|k-1}^{(i)}\right)
$$

バイアスの更新量は：

$$
b_{k|k}^{(i)} = b_{k|k-1}^{(i)} + K_k^{(b)} \left(z_k^{(i)} - x_{k|k-1}^{(i)}\right)
$$

**直感的な解釈**：
- LSTM予測値 $x_{k|k-1}$ が観測値 $z_k$ より **系統的に低い** → イノベーション $(z_k - x_{k|k-1})$ が正 → $b_k$ が正方向に更新される
- LSTM予測値が観測値より **系統的に高い** → $b_k$ が負方向に更新される
- $K_k^{(b)}$ は $\mathbf{P}_{xy}$ の第2成分（バイアスと観測の共分散）に比例するため、バイアスと観測の「関連の強さ」に応じて学習速度が調整される

##### バイアス学習の収束イメージ

```
時刻 k=0:  b = 0.000   (初期値、無バイアス仮定)
時刻 k=10: b = +0.012  (LSTM が過小予測傾向を検出)
時刻 k=30: b = +0.021  (定常値に収束しつつある)
時刻 k=50: b = +0.019  (安定した補正量として機能)
```

アンサンブルメンバー間のバイアス分散が収束することで、推定の信頼度も同時に評価できます。

---

#### 参考：`ErrorPatternKalmanFilter` の系統誤差バイアス（旧方式）

旧方式（`ErrorPatternKalmanFilter`）では、バイアスをEnKFで学習するのではなく、**LSTM予測誤差の移動平均**で算出します：

$$
b_k = \frac{\sum_{j=k-W+1}^{k} w_j \cdot e_j}{\sum_j w_j} \times \alpha_{\text{gain}}
$$

- $e_j = z_j - x^{\text{LSTM}}_j$：過去の観測 − LSTM予測誤差
- $w_j = \text{linspace}(1.0,\, 2.0,\, W)$：直近ほど重みが大きい線形重み
- $W$：学習ウィンドウ幅（デフォルト10日）
- $\alpha_{\text{gain}}$：バイアス補正係数（初期段階 `0.05`、通常段階 `0.18`）

補正量に上限 $\pm\delta_{\max}$（デフォルト `0.03`）を設けて過剰補正を防ぎます。

| 項目 | AugmentedStateEnKF（現方式） | ErrorPatternKalmanFilter（旧方式） |
|------|-----|------|
| バイアス導出 | カルマン更新式（共分散ベース） | 誤差の移動加重平均 |
| 不確かさ評価 | アンサンブル分散で自動推定 | なし |
| 適応速度 | $K_k^{(b)}$ で自動調整 | 固定ゲインで調整 |
| 解釈性 | 確率的フレームワーク | シンプルで直感的 |

#### 利点

1. **明示的な補正関数不要**: 観測から直接バイアスを学習
2. **非線形に強い**: アンサンブル手法で非線形関係を表現
3. **適応的**: アンサンブル分散から誤差評価を自動更新
4. **解釈可能**: バイアスの進化過程をトラッキング


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

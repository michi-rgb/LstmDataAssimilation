"""
データ同化検証スクリプト
LSTMで予測 → カルマンフィルタで補正
直近3ヶ月分を1日ずつ追加しながら検証
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import sys
sys.path.insert(0, 'src')

from train_lstm import LSTMModel, create_sequences
from kalman_filter import KalmanFilter, AdaptiveKalmanFilter, ErrorPatternKalmanFilter, AugmentedStateEnKF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_data():
    """
    学習済みモデルと検証データを読み込み
    """
    print("========== モデルとデータを読み込み中 ==========")
    
    # モデルを読み込み
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    model.eval()
    print("✓ LSTMモデル読み込み完了: models/lstm_model.pth")
    
    # スケーラーを読み込み
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ スケーラー読み込み完了: models/scaler.pkl")
    
    # 学習データと検証データを読み込み
    df_train = pd.read_csv('data/train_data.csv', index_col='Date', parse_dates=True)
    df_test = pd.read_csv('data/test_data.csv', index_col='Date', parse_dates=True)
    
    print(f"\n学習データ: {len(df_train)}行")
    print(f"検証データ: {len(df_test)}行")
    
    return model, scaler, df_train, df_test

def predict_next(model, recent_sequence):
    """
    LSTMで次の日の値を予測
    Args:
        model: 学習済みLSTMモデル
        recent_sequence: (seq_length, 1) の配列
    Returns:
        予測値
    """
    X = torch.from_numpy(recent_sequence.reshape(1, -1, 1)).float().to(device)
    
    with torch.no_grad():
        output = model(X)
    
    return output.item()

def data_assimilation_experiment():
    """
    データ同化実験
    1. 学習済みLSTMで次の日の株価を予測
    2. 実績値が得られたら、カルマンフィルタで更新（誤差学習に使用）
    3. 予測値を主指標、補正値を副指標として実値で評価
    """
    print("\n========== データ同化実験開始 ==========\n")
    
    model, scaler, df_train, df_test = load_model_and_data()
    
    # 最新の学習データを取得
    seq_length = 30
    train_data = df_train['Close'].values.reshape(-1, 1)
    recent_data = train_data[-seq_length:].copy()
    
    print(f"初期シーケンス: {recent_data[-1, 0]:.6f} (正規化済み)")
    
    # 誤差パターン学習付きカルマンフィルタを初期化
    # 予測ステップでLSTM誤差系列を学習し、系統バイアスを補正
    kf = ErrorPatternKalmanFilter(
        process_variance=0.00028,  # Q: 追従性を少し強化
        measurement_variance=0.00006,  # R: 観測信頼を少し強化
        initial_value=recent_data[-1, 0],  # 初期値: 最新の正規化済み株価
        initial_estimate_error=0.001,  # 初期推定誤差分散
        error_history_window=10,  # 誤差学習ウィンドウ
        bias_gain_early=0.06,
        bias_gain_main=0.22,
        max_bias_correction=0.035
    )
    
    print(f"カルマンフィルタ初期化:")
    print(f"  Q (システムノイズ分散): {kf.Q}")
    print(f"  R (観測ノイズ分散): {kf.R}\n")
    
    # 結果を保存するリスト
    results = []
    
    test_data = df_test['Close'].values
    test_dates = df_test.index
    
    print(f"検証期間: {len(df_test)}日\n")
    print(f"{'日付':<10} {'実績値':>9} {'LSTM予測値':>9} {'KF予測値':>9} {'KF補正値':>9} {'LSTM誤差':>8} {'KF予測誤差':>10} {'KF補正誤差':>10}")
    print("-" * 116)
    
    # シーケンシャルに検証
    for day_idx in range(len(test_data)):
        current_date = test_dates[day_idx].strftime("%Y-%m-%d")
        actual_normalized = test_data[day_idx]
        
        # LSTMで予測
        lstm_pred = predict_next(model, recent_data)

        # カルマンフィルタで更新（内部で予測→更新を実行）
        kf_corrected = kf.update(actual_normalized, lstm_pred)
        x_k_pred = kf.history_predicted[-1] if kf.history_predicted else lstm_pred
        
        # 逆正規化して実値に戻す
        actual_value = scaler.inverse_transform([[actual_normalized]])[0, 0]
        lstm_pred_value = scaler.inverse_transform([[lstm_pred]])[0, 0]
        kf_pred_value = scaler.inverse_transform([[x_k_pred]])[0, 0]
        kf_corrected_value = scaler.inverse_transform([[kf_corrected]])[0, 0]
        
        # 誤差を計算（予測誤差を主指標とする）
        lstm_error = abs(actual_value - lstm_pred_value)
        kf_pred_error = abs(actual_value - kf_pred_value)
        kf_corrected_error = abs(actual_value - kf_corrected_value)
        
        # 結果を保存
        results.append({
            'date': current_date,
            'actual_normalized': actual_normalized,
            'lstm_pred_normalized': lstm_pred,
            'kf_pred_normalized': x_k_pred,
            'kf_corrected_normalized': kf_corrected,
            'actual_value': actual_value,
            'lstm_pred_value': lstm_pred_value,
            'kf_pred_value': kf_pred_value,
            'kf_corrected_value': kf_corrected_value,
            'lstm_error': lstm_error,
            'kf_pred_error': kf_pred_error,
            'kf_corrected_error': kf_corrected_error,
            'kf_error': kf_corrected_error,
            'bias_correction': kf.history_bias_correction[-1] if hasattr(kf, 'history_bias_correction') and kf.history_bias_correction else 0.0,
            'kf_gain': kf.history_gain[-1] if kf.history_gain else 0,
            'kf_error_cov': kf.history_error_cov[-1] if kf.history_error_cov else 0
        })
        
        # 最新のシーケンスを更新（実績値を使用）
        recent_data = np.vstack([recent_data[1:], [[actual_normalized]]])
        
        # 結果を表示
        print(f"{current_date:<12} {actual_value:>12.2f} {lstm_pred_value:>12.2f} {kf_pred_value:>12.2f} {kf_corrected_value:>12.2f} {lstm_error:>10.2f} {kf_pred_error:>12.2f} {kf_corrected_error:>12.2f}")
    
    # 結果をDataFrameに変換して保存
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/data_assimilation_results.csv', index=False)
    print(f"\n✓ 結果を保存: results/data_assimilation_results.csv")
    
    # 精度指標を計算
    print(f"\n========== 評価指標 ==========")
    
    # MAE (Mean Absolute Error)
    lstm_mae = results_df['lstm_error'].mean()
    kf_pred_mae = results_df['kf_pred_error'].mean()
    kf_corrected_mae = results_df['kf_corrected_error'].mean()
    
    print(f"MAE（平均絶対誤差）:")
    print(f"  LSTM: {lstm_mae:.2f} 円")
    print(f"  カルマンフィルタ予測（主指標）: {kf_pred_mae:.2f} 円")
    print(f"  改善率（主指標）: {(1 - kf_pred_mae/lstm_mae)*100:.2f}%")
    print(f"  カルマンフィルタ補正（副指標）: {kf_corrected_mae:.2f} 円")
    print(f"  改善率（副指標）: {(1 - kf_corrected_mae/lstm_mae)*100:.2f}%")
    
    # RMSE (Root Mean Square Error)
    lstm_rmse = np.sqrt((results_df['lstm_error']**2).mean())
    kf_pred_rmse = np.sqrt((results_df['kf_pred_error']**2).mean())
    kf_corrected_rmse = np.sqrt((results_df['kf_corrected_error']**2).mean())
    
    print(f"\nRMSE（二乗平均平方根誤差）:")
    print(f"  LSTM: {lstm_rmse:.2f} 円")
    print(f"  カルマンフィルタ予測（主指標）: {kf_pred_rmse:.2f} 円")
    print(f"  改善率（主指標）: {(1 - kf_pred_rmse/lstm_rmse)*100:.2f}%")
    print(f"  カルマンフィルタ補正（副指標）: {kf_corrected_rmse:.2f} 円")
    print(f"  改善率（副指標）: {(1 - kf_corrected_rmse/lstm_rmse)*100:.2f}%")
    
    # カルマンゲインの統計
    avg_gain = np.mean(results_df['kf_gain'])
    print(f"\nカルマンゲイン統計:")
    print(f"  平均: {avg_gain:.6f}")
    print(f"  最小: {results_df['kf_gain'].min():.6f}")
    print(f"  最大: {results_df['kf_gain'].max():.6f}")
    
    # 最終推定誤差共分散
    print(f"\n最終推定誤差共分散: {results_df['kf_error_cov'].iloc[-1]:.8f}")
    
    return results_df


def data_assimilation_augmented_enkf_experiment():
    """
    拡張状態EnKFによるデータ同化実験
    LSTM予測値に学習中のバイアスを加え、観測前に補正した上で観測を同化する。
    """
    print("\n========== 拡張状態EnKFデータ同化実験開始 ==========")

    model, scaler, df_train, df_test = load_model_and_data()

    seq_length = 30
    train_data = df_train['Close'].values.reshape(-1, 1)
    recent_data = train_data[-seq_length:].copy()

    enkf = AugmentedStateEnKF(
        ensemble_size=40,
        process_variance_x=0.00025,
        process_variance_bias=0.000012,
        measurement_variance=0.00008,
        initial_state=recent_data[-1, 0],
        initial_bias=0.0,
        initial_state_spread=0.003,
        initial_bias_spread=0.008,
    )

    print(f"EnKF初期化: ensemble_size={enkf.N}, Q_x={enkf.Q_x}, Q_b={enkf.Q_b}, R={enkf.R}\n")

    results = []
    test_data = df_test['Close'].values
    test_dates = df_test.index

    print(f"検証期間: {len(df_test)}日\n")
    print(f"{'日付':<10} {'実績値':>9} {'LSTM予測値':>9} {'EnKF補正予測':>11} {'EnKF更新値':>11} {'LSTM誤差':>8} {'EnKF予測誤差':>10} {'EnKF更新誤差':>10}")
    print("-" * 108)

    for day_idx in range(len(test_data)):
        current_date = test_dates[day_idx].strftime("%Y-%m-%d")
        actual_normalized = test_data[day_idx]

        lstm_pred = predict_next(model, recent_data)
        enkf_prior, _ = enkf.predict(lstm_pred)
        enkf_analysis, _ = enkf.update(actual_normalized)

        actual_value = scaler.inverse_transform([[actual_normalized]])[0, 0]
        lstm_pred_value = scaler.inverse_transform([[lstm_pred]])[0, 0]
        enkf_prior_value = scaler.inverse_transform([[enkf_prior]])[0, 0]
        enkf_analysis_value = scaler.inverse_transform([[enkf_analysis]])[0, 0]

        lstm_error = abs(actual_value - lstm_pred_value)
        enkf_prior_error = abs(actual_value - enkf_prior_value)
        enkf_analysis_error = abs(actual_value - enkf_analysis_value)

        results.append({
            'date': current_date,
            'actual_normalized': actual_normalized,
            'lstm_pred_normalized': lstm_pred,
            'kf_pred_normalized': enkf_prior,
            'kf_corrected_normalized': enkf_analysis,
            'actual_value': actual_value,
            'lstm_pred_value': lstm_pred_value,
            'kf_pred_value': enkf_prior_value,
            'kf_corrected_value': enkf_analysis_value,
            'lstm_error': lstm_error,
            'kf_pred_error': enkf_prior_error,
            'kf_corrected_error': enkf_analysis_error,
            'kf_error': enkf_analysis_error,
            'kf_gain': enkf.history_gain[-1] if enkf.history_gain else 0,
            'kf_error_cov': enkf.history_error_cov[-1] if enkf.history_error_cov else 0,
        })

        recent_data = np.vstack([recent_data[1:], [[actual_normalized]]])

        print(f"{current_date:<12} {actual_value:>12.2f} {lstm_pred_value:>12.2f} {enkf_prior_value:>12.2f} {enkf_analysis_value:>12.2f} {lstm_error:>10.2f} {enkf_prior_error:>12.2f} {enkf_analysis_error:>12.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/data_assimilation_results.csv', index=False)
    print(f"\n✓ 結果を保存: results/data_assimilation_results.csv")

    print("\n========== 評価指標 ==========")
    lstm_mae = results_df['lstm_error'].mean()
    kf_pred_mae = results_df['kf_pred_error'].mean()
    kf_corrected_mae = results_df['kf_corrected_error'].mean()

    print(f"MAE（平均絶対誤差）:")
    print(f"  LSTM: {lstm_mae:.2f} 円")
    print(f"  EnKF補正予測: {kf_pred_mae:.2f} 円")
    print(f"  改善率: {(1 - kf_pred_mae/lstm_mae)*100:.2f}%")
    print(f"  EnKF更新値: {kf_corrected_mae:.2f} 円")
    print(f"  改善率: {(1 - kf_corrected_mae/lstm_mae)*100:.2f}%")

    lstm_rmse = np.sqrt((results_df['lstm_error']**2).mean())
    kf_pred_rmse = np.sqrt((results_df['kf_pred_error']**2).mean())
    kf_corrected_rmse = np.sqrt((results_df['kf_corrected_error']**2).mean())

    print(f"\nRMSE（二乗平均平方根誤差）:")
    print(f"  LSTM: {lstm_rmse:.2f} 円")
    print(f"  EnKF補正予測: {kf_pred_rmse:.2f} 円")
    print(f"  改善率: {(1 - kf_pred_rmse/lstm_rmse)*100:.2f}%")
    print(f"  EnKF更新値: {kf_corrected_rmse:.2f} 円")
    print(f"  改善率: {(1 - kf_corrected_rmse/lstm_rmse)*100:.2f}%")

    avg_gain = np.mean(results_df['kf_gain'])
    print(f"\nカルマンゲイン統計:")
    print(f"  平均: {avg_gain:.6f}")
    print(f"  最小: {results_df['kf_gain'].min():.6f}")
    print(f"  最大: {results_df['kf_gain'].max():.6f}")
    print(f"\n最終推定誤差共分散: {results_df['kf_error_cov'].iloc[-1]:.8f}")

    return results_df

if __name__ == "__main__":
    results_df = data_assimilation_augmented_enkf_experiment()

    import visualize_results
    visualize_results.visualize_results()
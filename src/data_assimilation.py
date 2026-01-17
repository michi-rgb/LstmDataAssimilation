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
from kalman_filter import KalmanFilter, AdaptiveKalmanFilter

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
    2. 実績値が得られたら、カルマンフィルタで補正
    3. 補正値をスケーラーで逆正規化して実値で評価
    """
    print("\n========== データ同化実験開始 ==========\n")
    
    model, scaler, df_train, df_test = load_model_and_data()
    
    # 最新の学習データを取得
    seq_length = 30
    train_data = df_train['Close'].values.reshape(-1, 1)
    recent_data = train_data[-seq_length:].copy()
    
    print(f"初期シーケンス: {recent_data[-1, 0]:.6f} (正規化済み)")
    
    # カルマンフィルタを初期化
    # システムノイズ（予測誤差）と観測ノイズ（実績と予測の誤差）の分散を設定
    kf = KalmanFilter(
        process_variance=0.0002,  # Q: システムノイズ分散
        measurement_variance=0.0001,  # R: 観測ノイズ分散
        initial_value=recent_data[-1, 0],  # 初期値: 最新の正規化済み株価
        initial_estimate_error=0.001  # 初期推定誤差分散
    )
    
    print(f"カルマンフィルタ初期化:")
    print(f"  Q (システムノイズ分散): {kf.Q}")
    print(f"  R (観測ノイズ分散): {kf.R}\n")
    
    # 結果を保存するリスト
    results = []
    
    test_data = df_test['Close'].values
    test_dates = df_test.index
    
    print(f"検証期間: {len(df_test)}日\n")
    print(f"{'日付':<10} {'実績値':>9} {'LSTM予測値':>9} {'KF予測値':>9} {'KF補正値':>9} {'LSTM予測誤差':>8} {'KF補正誤差':>8}") # 全角は2文字分
    print("-" * 90)
    
    # シーケンシャルに検証
    for day_idx in range(len(test_data)):
        current_date = test_dates[day_idx].strftime("%Y-%m-%d")
        actual_normalized = test_data[day_idx]
        
        # LSTMで予測
        lstm_pred = predict_next(model, recent_data)

        # カルマンフィルタで予測
        x_k_pred, P_k_pred = kf.predict(lstm_pred)
        
        # カルマンフィルタで補正
        kf_corrected = kf.update(actual_normalized, lstm_pred)
        
        # 逆正規化して実値に戻す
        actual_value = scaler.inverse_transform([[actual_normalized]])[0, 0]
        lstm_pred_value = scaler.inverse_transform([[lstm_pred]])[0, 0]
        kf_pred_value = scaler.inverse_transform([[x_k_pred]])[0, 0]
        kf_corrected_value = scaler.inverse_transform([[kf_corrected]])[0, 0]
        
        # 誤差を計算
        lstm_error = abs(actual_value - lstm_pred_value)
        kf_error = abs(actual_value - kf_corrected_value)
        
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
            'kf_error': kf_error,
            'kf_gain': kf.history_gain[-1] if kf.history_gain else 0,
            'kf_error_cov': kf.history_error_cov[-1] if kf.history_error_cov else 0
        })
        
        # 最新のシーケンスを更新（実績値を使用）
        recent_data = np.vstack([recent_data[1:], [[actual_normalized]]])
        
        # 結果を表示
        print(f"{current_date:<12} {actual_value:>12.2f} {lstm_pred_value:>12.2f} {kf_pred_value:>12.2f} {kf_corrected_value:>12.2f} {lstm_error:>12.2f} {kf_error:>12.2f}")
    
    # 結果をDataFrameに変換して保存
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/data_assimilation_results.csv', index=False)
    print(f"\n✓ 結果を保存: results/data_assimilation_results.csv")
    
    # 精度指標を計算
    print(f"\n========== 評価指標 ==========")
    
    # MAE (Mean Absolute Error)
    lstm_mae = results_df['lstm_error'].mean()
    kf_mae = results_df['kf_error'].mean()
    
    print(f"MAE（平均絶対誤差）:")
    print(f"  LSTM: {lstm_mae:.2f} 円")
    print(f"  カルマンフィルタ補正: {kf_mae:.2f} 円")
    print(f"  改善率: {(1 - kf_mae/lstm_mae)*100:.2f}%")
    
    # RMSE (Root Mean Square Error)
    lstm_rmse = np.sqrt((results_df['lstm_error']**2).mean())
    kf_rmse = np.sqrt((results_df['kf_error']**2).mean())
    
    print(f"\nRMSE（二乗平均平方根誤差）:")
    print(f"  LSTM: {lstm_rmse:.2f} 円")
    print(f"  カルマンフィルタ補正: {kf_rmse:.2f} 円")
    print(f"  改善率: {(1 - kf_rmse/lstm_rmse)*100:.2f}%")
    
    # カルマンゲインの統計
    avg_gain = np.mean(results_df['kf_gain'])
    print(f"\nカルマンゲイン統計:")
    print(f"  平均: {avg_gain:.6f}")
    print(f"  最小: {results_df['kf_gain'].min():.6f}")
    print(f"  最大: {results_df['kf_gain'].max():.6f}")
    
    # 最終推定誤差共分散
    print(f"\n最終推定誤差共分散: {results_df['kf_error_cov'].iloc[-1]:.8f}")
    
    return results_df

if __name__ == "__main__":
    results_df = data_assimilation_experiment()

    import visualize_results
    visualize_results.visualize_results()
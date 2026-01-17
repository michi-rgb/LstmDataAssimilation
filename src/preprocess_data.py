"""
データ前処理スクリプト
学習用データ（2016/1/12～2025/10/11）と検証用データ（2025/10/12～2026/1/12）に分割
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def load_and_preprocess_data():
    """
    CSVからデータを読み込み、前処理を実施
    """
    print("データを読み込み中...")
    
    # CSVを読み込み
    df = pd.read_csv("data/nikkei_raw.csv", index_col='Date', parse_dates=True)
    
    # NaNを削除
    df = df.dropna()
    
    print(f"✓ データ読み込み完了: {len(df)}行")
    print(f"  期間: {df.index[0].date()} - {df.index[-1].date()}")
    
    # 学習用データの期間: 2016/1/12～2025/10/11
    train_start = datetime(2016, 1, 12)
    train_end = datetime(2025, 10, 11)
    
    # 検証用データの期間: 2025/10/12～2026/1/12
    test_start = datetime(2025, 10, 12)
    test_end = datetime(2026, 1, 12)
    
    # データを分割
    df_train = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
    df_test = df.loc[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    print(f"\n学習用データ:")
    print(f"  期間: {df_train.index[0].date()} - {df_train.index[-1].date()}")
    print(f"  データポイント数: {len(df_train)}")
    
    print(f"\n検証用データ:")
    print(f"  期間: {df_test.index[0].date()} - {df_test.index[-1].date()}")
    print(f"  データポイント数: {len(df_test)}")
    
    # MinMaxスケーラーで正規化（学習データの統計量を使用）
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 学習データで fitして、train/testの両方を正規化
    scaler.fit(df_train[['Close']])
    
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled['Close'] = scaler.transform(df_train[['Close']])
    df_test_scaled['Close'] = scaler.transform(df_test[['Close']])
    
    # スケーラーを保存
    import pickle
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✓ スケーラー保存: models/scaler.pkl")
    
    # データを保存
    df_train_scaled.to_csv('data/train_data.csv')
    df_test_scaled.to_csv('data/test_data.csv')
    
    print(f"✓ 前処理済みデータを保存:")
    print(f"  data/train_data.csv")
    print(f"  data/test_data.csv")
    
    return df_train_scaled, df_test_scaled, scaler

if __name__ == "__main__":
    df_train, df_test, scaler = load_and_preprocess_data()

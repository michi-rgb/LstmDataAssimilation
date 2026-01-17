"""
日経平均株価データをYahooFinanceからダウンロードするスクリプト
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def download_nikkei_data():
    """
    日経平均株価データを過去10年分ダウンロード
    """
    print("日経平均株価データをダウンロード中...")
    
    # 現在日時から10年前までのデータを取得
    end_date = datetime(2026, 1, 12)
    start_date = end_date - timedelta(days=365*10)
    
    # 日経平均のティッカー
    nikkei_ticker = "^N225"
    
    try:
        # データをダウンロード
        data = yf.download(nikkei_ticker, start=start_date, end=end_date)
        
        # マルチインデックスの場合、Closeカラムを抽出
        if isinstance(data.columns, pd.MultiIndex):
            data_close = data['Close'].copy()
        else:
            data_close = data[['Close']].copy()
        
        # Series を DataFrame に変換
        if isinstance(data_close, pd.Series):
            data_close = data_close.to_frame()
            data_close.columns = ['Close']
        else:
            data_close.columns = ['Close']
        
        data_close.index.name = 'Date'
        
        # CSVに保存
        output_path = "data/nikkei_raw.csv"
        data_close.to_csv(output_path)
        print(f"✓ データ保存完了: {output_path}")
        print(f"  期間: {data_close.index[0].date()} - {data_close.index[-1].date()}")
        print(f"  データポイント数: {len(data_close)}")
        
        return data_close
        
    except Exception as e:
        print(f"✗ ダウンロード失敗: {e}")
        return None

if __name__ == "__main__":
    download_nikkei_data()

"""
LSTM モデルの定義と学習
PyTorchを使用
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# 日本語フォントの設定
rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class LSTMModel(nn.Module):
    """
    LSTM予測モデル
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            y: (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 最後のタイムステップの出力を使用
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        y = self.fc(last_out)
        return y

def create_sequences(data, seq_length=30):
    """
    時系列データからシーケンスを生成
    Args:
        data: (n_samples, 1) の配列
        seq_length: シーケンス長
    Returns:
        X: (n_sequences, seq_length, 1)
        y: (n_sequences, 1)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_lstm():
    """
    LSTMモデルを学習
    """
    print("\n========== LSTM学習開始 ==========")
    
    # データを読み込み
    df_train = pd.read_csv('data/train_data.csv', index_col='Date', parse_dates=True)
    
    data = df_train['Close'].values.reshape(-1, 1)
    print(f"学習データシェイプ: {data.shape}")
    
    # シーケンスを生成
    seq_length = 30  # 30日間のデータで次の日を予測
    X_train, y_train = create_sequences(data, seq_length)
    
    print(f"シーケンス数: {len(X_train)}")
    print(f"X_train シェイプ: {X_train.shape}")
    print(f"y_train シェイプ: {y_train.shape}")
    
    # テンソルに変換
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    
    # モデルを初期化
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    
    # 損失関数と最適化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 学習ループ
    num_epochs = 100
    batch_size = 32
    
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # ミニバッチ学習
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            # 順伝播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print(f"\n✓ 学習完了")
    print(f"  最終損失: {loss_history[-1]:.6f}")
    
    # モデルを保存
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    print(f"✓ モデル保存: models/lstm_model.pth")
    
    # 学習曲線をcsvに保存
    loss_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'loss': loss_history})
    loss_df.to_csv('results/training_loss.csv', index=False)
    print(f"✓ 学習曲線保存: results/training_loss.csv")
    
    # 学習曲線グラフを保存
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(loss_df['epoch'], loss_df['loss'], 'b-', linewidth=2, label='損失')
    ax.set_xlabel('エポック', fontsize=12)
    ax.set_ylabel('損失（MSE、対数スケール）', fontsize=12)
    ax.set_title('LSTM学習曲線', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    output_file = 'results/training_loss.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ グラフ保存: {output_file}")
    plt.close()
    
    return model, loss_history

if __name__ == "__main__":
    model, loss_history = train_lstm()

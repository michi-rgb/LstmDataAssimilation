"""
結果可視化スクリプト
LSTM予測とカルマンフィルタ予測/補正をグラフで表示
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 日本語フォントの設定
rcParams['font.sans-serif'] = ['Yu Gothic', 'Hiragino Sans', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def visualize_results():
    """
    データ同化結果を可視化
    """
    print("結果を可視化中...")
    
    # 結果データを読み込み
    results_df = pd.read_csv('results/data_assimilation_results.csv')
    training_loss_df = pd.read_csv('results/training_loss.csv')
    
    # 日付をDatetimeに変換
    results_df['date'] = pd.to_datetime(results_df['date'])
    
    x = np.arange(len(results_df))
    tick_indices = np.linspace(0, len(results_df)-1, 6, dtype=int)
    
    # ========== グラフ1: 学習曲線 ==========
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(training_loss_df['epoch'], training_loss_df['loss'], 'b-', linewidth=2, label='損失')
    ax1.set_xlabel('エポック', fontsize=12)
    ax1.set_ylabel('損失（MSE）', fontsize=12)
    ax1.set_title('LSTM学習曲線', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    plt.tight_layout()
    
    output_file1 = 'results/training_loss.png'
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file1}")
    plt.close()
    
    # ========== グラフ2: 予測値 vs 実績値 vs 補正値 ==========
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    
    ax2.plot(x, results_df['actual_value'], 'o-', label='実績値', 
             linewidth=2, markersize=5, color='black', zorder=3)
    ax2.plot(x, results_df['lstm_pred_value'], 'o-', label='LSTM予測値', 
             linewidth=1.5, markersize=4, color='red', alpha=0.7, zorder=2)
    ax2.plot(x, results_df['kf_pred_value'], 'o-', label='カルマンフィルタ予測値', 
             linewidth=1.5, markersize=4, color='orange', alpha=0.7, zorder=2)
    ax2.plot(x, results_df['kf_corrected_value'], 'o-', label='カルマンフィルタ補正値', 
             linewidth=1.5, markersize=4, color='green', alpha=0.7, zorder=2)
    
    ax2.set_xlabel('日数', fontsize=12)
    ax2.set_ylabel('株価（円）', fontsize=12)
    ax2.set_title('LSTM予測値 vs カルマンフィルタ補正値 vs 実績値', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    # x軸にいくつかの日付ラベルを表示
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([results_df['date'].iloc[i].strftime('%m-%d') for i in tick_indices], 
                        rotation=45)
    plt.tight_layout()
    
    output_file2 = 'results/prediction_comparison.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file2}")
    plt.close()
    
    # ========== グラフ3: 予測誤差比較（主指標） ==========
    fig3, ax3 = plt.subplots(figsize=(14, 5))

    bar_width = 0.4
    ax3.bar(x - bar_width/2, results_df['lstm_error'], width=bar_width,
            color='red', alpha=0.6, label='LSTM予測誤差')
    ax3.bar(x + bar_width/2, results_df['kf_pred_error'], width=bar_width,
            color='orange', alpha=0.6, label='KF予測誤差（主指標）')

    if 'kf_corrected_error' in results_df.columns:
        ax3.plot(x, results_df['kf_corrected_error'], color='green', linewidth=1.8,
                 label='KF補正誤差（副指標）', alpha=0.9)

    ax3.set_xlabel('日数', fontsize=12)
    ax3.set_ylabel('絶対誤差（円）', fontsize=12)
    ax3.set_title('予測誤差の比較（LSTM vs KF予測）', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='best')

    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels([results_df['date'].iloc[i].strftime('%m-%d') for i in tick_indices],
                        rotation=45)
    plt.tight_layout()

    output_file3 = 'results/prediction_error_comparison.png'
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file3}")
    plt.close()

    # ========== グラフ3b: 予測差分比較（符号付き） ==========
    # 実績 - 予測（正: 過小予測 / 負: 過大予測）
    fig3b, ax3b = plt.subplots(figsize=(14, 5))

    lstm_signed_diff = results_df['actual_value'] - results_df['lstm_pred_value']
    kf_pred_signed_diff = results_df['actual_value'] - results_df['kf_pred_value']

    bar_width_signed = 0.4
    ax3b.bar(x - bar_width_signed/2, lstm_signed_diff, width=bar_width_signed,
             color='red', alpha=0.6, label='LSTM差分（実績-予測）')
    ax3b.bar(x + bar_width_signed/2, kf_pred_signed_diff, width=bar_width_signed,
             color='orange', alpha=0.6, label='KF予測差分（実績-予測）')

    if 'kf_corrected_value' in results_df.columns:
        kf_corrected_signed_diff = results_df['actual_value'] - results_df['kf_corrected_value']
        ax3b.plot(x, kf_corrected_signed_diff, color='green', linewidth=1.8,
                  label='KF補正差分（副指標）', alpha=0.9)

    ax3b.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3b.set_xlabel('日数', fontsize=12)
    ax3b.set_ylabel('差分（円）', fontsize=12)
    ax3b.set_title('予測差分の比較（符号付き: 実績-予測）', fontsize=14, fontweight='bold')
    ax3b.grid(True, alpha=0.3)
    ax3b.legend(fontsize=10, loc='best')

    ax3b.set_xticks(tick_indices)
    ax3b.set_xticklabels([results_df['date'].iloc[i].strftime('%m-%d') for i in tick_indices],
                         rotation=45)
    plt.tight_layout()

    output_file3b = 'results/prediction_error_signed_comparison.png'
    plt.savefig(output_file3b, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file3b}")
    plt.close()

    # ========== グラフ4: カルマンゲイン ==========
    # カルマンゲイン（K_k）は、予測値と実績値のどちらをどの程度信頼するかを決める重み係数
    # K_k = P_k_pred / (P_k_pred + measurement_variance)
    # 予測が正確	小さい（0に近い）	予測値を信頼し、実績値による補正を少なくする
    # 観測が正確	大きい（1に近い）	実績値を信頼し、予測値を大きく補正する
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    ax4.plot(x, results_df['kf_gain'], 'o-', color='blue', linewidth=2, markersize=5)
    ax4.set_xlabel('日数', fontsize=12)
    ax4.set_ylabel('カルマンゲイン', fontsize=12)
    ax4.set_title('カルマンゲインの推移（観測値への信頼度, 予測値への不信頼度）', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=results_df['kf_gain'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'平均: {results_df["kf_gain"].mean():.4f}')
    ax4.legend(fontsize=10)
    
    # x軸ラベル
    ax4.set_xticks(tick_indices)
    ax4.set_xticklabels([results_df['date'].iloc[i].strftime('%m-%d') for i in tick_indices], 
                         rotation=45)
    ax4.set_ylim(0, 1)
    plt.tight_layout()
    
    output_file4 = 'results/kalman_gain.png'
    plt.savefig(output_file4, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file4}")
    plt.close()
    
    # ========== グラフ5: 推定誤差共分散 ==========
    # カルマンフィルタが現在の推定値がどの程度正確かを表す
    # 観測を繰り返すごとに更新される動的な値
    # 観測を重ねると通常は減少する（信頼度が上がる）
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    ax5.plot(x, results_df['kf_error_cov'], 'o-', color='purple', linewidth=2, markersize=5)
    ax5.set_xlabel('日数', fontsize=12)
    ax5.set_ylabel('推定誤差共分散', fontsize=12)
    ax5.set_title('カルマンフィルタの推定誤差共分散の推移', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # x軸ラベル
    ax5.set_xticks(tick_indices)
    ax5.set_xticklabels([results_df['date'].iloc[i].strftime('%m-%d') for i in tick_indices], 
                         rotation=45)
    plt.tight_layout()
    
    output_file5 = 'results/kalman_error_covariance.png'
    plt.savefig(output_file5, dpi=150, bbox_inches='tight')
    print(f"✓ グラフを保存: {output_file5}")
    plt.close()

if __name__ == "__main__":
    visualize_results()

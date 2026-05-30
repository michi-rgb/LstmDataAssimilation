"""
カルマンフィルタの実装
観測値補正版：予測値と実績値の誤差のみを補正
"""
import numpy as np

class KalmanFilter:
    """
    スカラーカルマンフィルタ（1次元）
    観測値補正版：予測値を観測値で補正
    
    状態: x_k = 予測値
    観測: z_k = 実績値
    """
    
    def __init__(self, process_variance, measurement_variance, initial_value, initial_estimate_error):
        """
        Args:
            process_variance: プロセスノイズの分散 Q
            measurement_variance: 観測ノイズの分散 R
            initial_value: 初期値
            initial_estimate_error: 初期推定誤差分散
        """
        self.Q = process_variance  # プロセスノイズ分散
        self.R = measurement_variance  # 観測ノイズ分散
        
        self.x_k = initial_value  # 状態推定値
        self.P_k = initial_estimate_error  # 推定誤差分散
        
        # 履歴を保存
        self.history_predicted = []  # 予測値
        self.history_filtered = []   # 補正後の推定値
        self.history_gain = []       # カルマンゲイン
        self.history_error_cov = []  # 誤差共分散
        
    def predict(self, predicted_value):
        """
        予測ステップ
        Args:
            predicted_value: LSTMの予測値
        Returns:
            予測値
        """
        # 予測誤差共分散の更新: P_k|k-1 = P_k-1 + Q
        P_k_pred = self.P_k + self.Q
        
        # 予測値（状態方程式が単純なので予測値 = LSTMの出力）
        # x_k_pred = predicted_value
        # 予測値：前回のフィルタ値を基準に、LSTMの予測値で更新
        # x_k|k-1 = x_k-1|k-1 + (predicted_value - x_k-1|k-1) の重み付け
        # 実質LSTM予測値（今回の赤）と前回のフィルタ補正値（前回の緑）の中間が今回のフィルタ予測値（今回の橙）
        x_k_pred = self.x_k + 0.5 * (predicted_value - self.x_k)  # 重み0.5は調整可能
        
        
        self.history_predicted.append(x_k_pred)
        
        return x_k_pred, P_k_pred
    
    def update(self, measured_value, predicted_value):
        """
        更新（補正）ステップ
        Args:
            measured_value: 実際の観測値（実績値）
            predicted_value: LSTMの予測値
        Returns:
            補正後の推定値
        """
        # 予測ステップ
        x_k_pred, P_k_pred = self.predict(predicted_value)
        
        # カルマンゲインの計算: K_k = P_k|k-1 / (P_k|k-1 + R)
        K_k = P_k_pred / (P_k_pred + self.R)
        
        # 観測の誤差: y_k = z_k - x_k|k-1
        innovation = measured_value - x_k_pred
        
        # 状態推定値の更新: x_k|k = x_k|k-1 + K_k * (z_k - x_k|k-1)
        x_k = x_k_pred + K_k * innovation
        
        # 推定誤差分散の更新: P_k|k = (1 - K_k) * P_k|k-1
        P_k = (1.0 - K_k) * P_k_pred
        
        # 状態を更新
        self.x_k = x_k
        self.P_k = P_k
        
        # 履歴を保存
        self.history_filtered.append(x_k)
        self.history_gain.append(K_k)
        self.history_error_cov.append(P_k)
        
        return x_k
    
    def get_history(self):
        """
        フィルタリング履歴を取得
        """
        return {
            'predicted': np.array(self.history_predicted),
            'filtered': np.array(self.history_filtered),
            'gain': np.array(self.history_gain),
            'error_cov': np.array(self.history_error_cov)
        }

class ErrorPatternKalmanFilter(KalmanFilter):
    """
    改善版カルマンフィルタ：誤差パターン学習
    
    予測ステップでLSTMの系統的なバイアスを学習して補正
    実務的に使える翌日予測の精度を上げる
    """
    
    def __init__(
        self,
        process_variance,
        measurement_variance,
        initial_value,
        initial_estimate_error,
        error_history_window=10,
        bias_gain_early=0.03,
        bias_gain_main=0.12,
        max_bias_correction=0.02
    ):
        """
        Args:
            process_variance: プロセスノイズの分散 Q
            measurement_variance: 観測ノイズの分散 R
            initial_value: 初期値
            initial_estimate_error: 初期推定誤差分散
            error_history_window: 誤差パターン学習に使う直近日数（デフォルト10日）
            bias_gain_early: 初期学習段階のバイアス補正係数
            bias_gain_main: 通常学習段階のバイアス補正係数
            max_bias_correction: 1ステップの最大バイアス補正量（正規化空間）
        """
        super().__init__(process_variance, measurement_variance, initial_value, initial_estimate_error)
        
        self.error_history = []  # LSTM予測誤差の履歴
        self.error_history_window = error_history_window  # 学習ウィンドウサイズ
        self.history_bias_correction = []  # バイアス補正値の履歴
        self.bias_gain_early = bias_gain_early
        self.bias_gain_main = bias_gain_main
        self.max_bias_correction = max_bias_correction
    
    def predict(self, predicted_value):
        """
        改善版予測ステップ：過去の誤差パターンから補正
        
        Args:
            predicted_value: LSTMの予測値
        Returns:
            補正済み予測値, 予測誤差共分散
        """
        # 予測誤差共分散の更新: P_k|k-1 = P_k-1 + Q
        P_k_pred = self.P_k + self.Q
        
        # 直近の誤差パターンから、LSTMの系統的なバイアスを推定
        bias_correction = 0.0
        history_len = len(self.error_history)
        if history_len > 0:
            if history_len >= self.error_history_window:
                recent_errors = np.array(self.error_history[-self.error_history_window:])
                bias_gain = self.bias_gain_main
            else:
                recent_errors = np.array(self.error_history)
                bias_gain = self.bias_gain_early

            # 直近ほど重みを高くする指数重み平均で、古い誤差の影響を抑える
            weights = np.linspace(1.0, 2.0, len(recent_errors))
            bias = np.average(recent_errors, weights=weights)
            raw_correction = bias * bias_gain

            # 過剰補正を防ぐため、補正量に上限を設定
            bias_correction = float(np.clip(raw_correction, -self.max_bias_correction, self.max_bias_correction))
        
        # LSTM予測値のみを使用し、誤差パターンで補正
        x_k_pred = predicted_value - bias_correction
        
        # 履歴に保存
        self.history_predicted.append(x_k_pred)
        self.history_bias_correction.append(bias_correction)
        
        return x_k_pred, P_k_pred
    
    def update(self, measured_value, predicted_value):
        """
        更新（補正）ステップ：実績値が判明後に誤差パターンを記録
        
        Args:
            measured_value: 実際の観測値（実績値）
            predicted_value: LSTMの予測値
        Returns:
            補正後の推定値
        """
        # 予測ステップ
        x_k_pred, P_k_pred = self.predict(predicted_value)
        
        # カルマンゲインの計算: K_k = P_k|k-1 / (P_k|k-1 + R)
        K_k = P_k_pred / (P_k_pred + self.R)
        
        # 観測の誤差: y_k = z_k - x_k|k-1
        innovation = measured_value - x_k_pred
        
        # 状態推定値の更新: x_k|k = x_k|k-1 + K_k * (z_k - x_k|k-1)
        x_k = x_k_pred + K_k * innovation
        
        # 推定誤差分散の更新: P_k|k = (1 - K_k) * P_k|k-1
        P_k = (1.0 - K_k) * P_k_pred
        
        # 状態を更新
        self.x_k = x_k
        self.P_k = P_k
        
        # 履歴を保存
        self.history_filtered.append(x_k)
        self.history_gain.append(K_k)
        self.history_error_cov.append(P_k)
        
        # === 重要：LSTM予測の誤差パターンを記録 ===
        # predicted_value は元のLSTM予測値
        # measured_value は実績値
        # この誤差が次回の予測ステップで利用される
        lstm_error = measured_value - predicted_value
        self.error_history.append(lstm_error)
        
        return x_k
    
    def get_history(self):
        """
        フィルタリング履歴を取得
        """
        history = super().get_history()
        history['bias_correction'] = np.array(self.history_bias_correction)
        history['lstm_error'] = np.array(self.error_history)
        return history

class AdaptiveKalmanFilter(KalmanFilter):
    """
    適応カルマンフィルタ
    観測ノイズ分散を自動調整
    """
    
    def __init__(self, process_variance, measurement_variance, initial_value, initial_estimate_error):
        super().__init__(process_variance, measurement_variance, initial_value, initial_estimate_error)
        self.R_initial = measurement_variance
        self.innovation_history = []
    
    def update(self, measured_value, predicted_value):
        """
        適応更新：観測ノイズ分散を調整
        """
        x_k_pred, P_k_pred = self.predict(predicted_value)
        
        # 観測の誤差
        innovation = measured_value - x_k_pred
        self.innovation_history.append(innovation)
        
        # 直近10個のイノベーション（観測誤差）から分散を推定
        if len(self.innovation_history) > 10:
            recent_innovations = np.array(self.innovation_history[-10:])
            # 観測ノイズ分散を推定
            self.R = np.var(recent_innovations)
        
        # カルマンゲイン
        K_k = P_k_pred / (P_k_pred + self.R)
        
        # 状態推定値の更新
        x_k = x_k_pred + K_k * innovation
        
        # 推定誤差分散の更新
        P_k = (1.0 - K_k) * P_k_pred
        
        self.x_k = x_k
        self.P_k = P_k
        
        self.history_filtered.append(x_k)
        self.history_gain.append(K_k)
        self.history_error_cov.append(P_k)
        
        return x_k

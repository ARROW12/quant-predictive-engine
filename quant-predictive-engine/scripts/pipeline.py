import os
import json
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

# Configure industrial logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantEngine")

class QuantDataHandler:
    """Manages secure ingestion, deduplication, and archival of historical market data."""
    def __init__(self, ticker: str, storage_path: str):
        self.ticker = ticker
        self.storage_path = storage_path

    def extract_and_align(self) -> pd.DataFrame:
        logger.info(f"Ingesting intraday market data for {self.ticker}...")
        # Fetch 5 days of 5-minute bars to handle weekend gaps and guarantee overlap
        df_new = yf.download(tickers=self.ticker, period="5d", interval="5m", progress=False)
        if df_new.empty:
            logger.error("Market data ingestion returned empty payload.")
            return pd.DataFrame()
        
        # Normalize column mapping for yfinance multi-index outputs
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)
            
        df_new = df_new.reset_index()
        df_new['Datetime'] = pd.to_datetime(df_new['Datetime'])

        if os.path.exists(self.storage_path):
            df_old = pd.read_csv(self.storage_path)
            df_old['Datetime'] = pd.to_datetime(df_old['Datetime'])
            # Atomic deduplication via timestamp primary key
            df_total = pd.concat([df_old, df_new]).drop_duplicates(subset=['Datetime'])
            df_total = df_total.sort_values('Datetime').reset_index(drop=True)
        else:
            df_total = df_new

        # Constrain data lake size to prevent GitHub repository bloat (~1 year of 5m bars)
        df_total = df_total.tail(25000)
        df_total.to_csv(self.storage_path, index=False)
        logger.info(f"Data Lake synchronized. Total records: {len(df_total)}")
        return df_total


class FeatureEngineer:
    """Generates mathematically sound, stationary predictive features from raw market bars."""
    @staticmethod
    def construct(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Microstructure Signal: Log Returns (Stationary transformation)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Momentum Signal: Rolling Volatility-Adjusted Momentum
        df['rolling_mean_20'] = df['log_return'].rolling(window=20).mean()
        df['rolling_std_20'] = df['log_return'].rolling(window=20).std()
        
        # 3. Mean Reversion Signal: Price Distance from Moving Average (Z-Score)
        price_ma = df['Close'].rolling(window=20).mean()
        price_std = df['Close'].rolling(window=20).std()
        df['z_score_price'] = (df['Close'] - price_ma) / (price_std + 1e-8)
        
        # 4. Volume Dynamics: Ratio of short-term volume over long-term volume
        df['volume_shifter'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-8)
        
        # Target Variable: Predict the NEXT 5-minute log return
        df['target'] = df['log_return'].shift(-1)
        
        # Forward fill clean up and dropping edge artifacts
        df = df.ffill().dropna().reset_index(drop=True)
        return df


class QuantitativeBacktester:
    """Evaluates strategy health out-of-sample using institutional risk metrics."""
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        # Simple execution simulation: Go long if predicted return > 0, short if < 0
        strategy_returns = np.sign(y_pred) * y_true
        
        # Calculate Risk Metrics
        win_rate = float(np.sum(strategy_returns > 0) / len(strategy_returns))
        
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        # Annualized Sharpe Ratio for 5-minute bars assuming standard market hours
        sharpe = float((mean_ret / (std_ret + 1e-8)) * np.sqrt(252 * 78)) 
        
        cum_returns = np.exp(np.cumsum(strategy_returns))
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = float(np.min(drawdown))
        
        return {
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2)
        }


class ProductionPipeline:
    """Orchestrates execution workflow from extraction through front-end payload delivery."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.storage_dir = "data"
        self.data_path = os.path.join(self.storage_dir, "data.csv")
        self.payload_path = os.path.join(self.storage_dir, "predictions.json")
        
        os.makedirs(self.storage_dir, exist_ok=True)
        self.handler = QuantDataHandler(self.ticker, self.data_path)

    def execute(self):
        raw_data = self.handler.extract_and_align()
        if raw_data.empty or len(raw_data) < 100:
            logger.error("Insufficient historical structural data to execute modeling.")
            return
            
        feature_df = FeatureEngineer.construct(raw_data)
        
        feature_cols = ['log_return', 'rolling_mean_20', 'rolling_std_20', 'z_score_price', 'volume_shifter']
        X = feature_df[feature_cols]
        y = feature_df['target']
        
        # Chronological Out-of-Sample Validation Split (No random shuffling for time-series)
        split_idx = int(len(feature_df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # High-performance gradient boosting optimized for noisy financial series
        model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model performance out-of-sample
        test_preds = model.predict(X_test)
        risk_metrics = QuantitativeBacktester.evaluate(y_test.values, test_preds)
        
        # Extract tail vector to predict immediate upcoming horizon
        latest_vector = X.iloc[[-1]]
        next_log_return_pred = float(model.predict(latest_vector)[0])
        current_close = float(raw_data['Close'].iloc[-1])
        predicted_next_close = float(current_close * np.exp(next_log_return_pred))
        
        # Compile standardized JSON payload for static UI reading
        payload = {
            "meta": {
                "ticker": self.ticker,
                "execution_timestamp_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                "model_version": "HistGBM-v2.0"
            },
            "backtest_performance": risk_metrics,
            "signal": {
                "current_spot_price": round(current_close, 2),
                "predicted_target_price": round(predicted_close, 2),
                "expected_log_return": round(next_log_return_pred, 5),
                "direction": "BULLISH" if next_log_return_pred > 0 else "BEARISH"
            }
        }
        
        with open(self.payload_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info("Production payload compiled and dispatched to disk.")

if __name__ == "__main__":
    # Configure your trading asset here
    Pipeline = ProductionPipeline(ticker="AAPL")
    Pipeline.execute()
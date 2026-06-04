import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedAlphaEngine")

# 10 High-Liquidity F&O NSE Bluechips
INDIAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS", "MARUTI.NS"
]

class TechnicalIndicatorEngine:
    """Calculates stationary, vector-optimized technical features from raw OHLCV arrays."""
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Base Log Returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Native Relative Strength Index (RSI 14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. Native Average True Range (ATR 14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # 4. Native Moving Average Convergence Divergence (MACD 12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema_12 - ema_26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
        # 5. Price Distance from Rolling Structural Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['z_score_price'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-8)
        
        # Target Architecture: 3-bar forward cumulative log return (15-minute predictive runway)
        df['target'] = np.log(df['Close'].shift(-3) / df['Close'])
        
        return df.ffill().dropna().reset_index(drop=True)


def execute_alpha_pipeline(ticker: str, storage_path: str) -> dict:
    logger.info(f"Executing mathematical model layers for {ticker}...")
    df = yf.download(tickers=ticker, period="15d", interval="5m", progress=False)
    if df.empty or len(df) < 300:
        logger.warning(f"Insufficient matrix layout size for {ticker}. Skipping.")
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    processed_df = TechnicalIndicatorEngine.calculate_metrics(df)
    
    if len(processed_df) < 150:
        return None

    # Feature Matrix Layout Configuration
    feature_cols = ['log_return', 'rsi_14', 'atr_14', 'macd_hist', 'z_score_price']
    X = processed_df[feature_cols]
    y = processed_df['target']
    
    # Strictly Chronological Validation Train/Test Partition (80/20)
    split_idx = int(len(processed_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Regularized High-Performance Ensemble Model Configuration
    model = HistGradientBoostingRegressor(
        max_iter=75,
        learning_rate=0.02,
        max_leaf_nodes=15,
        l2_regularization=5.0,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Out-of-sample directional validation scoring
    test_predictions = model.predict(X_test)
    directional_accuracy = np.mean((test_predictions > 0) == (y_test.values > 0))
    
    # Compute current active market parameters
    latest_vector = X.iloc[[-1]]
    predicted_alpha = float(model.predict(latest_vector)[0])
    
    current_spot = float(processed_df['Close'].iloc[-1])
    current_atr = float(processed_df['atr_14'].iloc[-1])
    
    # Standard Options Chain Strike Step Alignment Mechanics
    strike_step = 50 if current_spot > 1000 else 10
    atm_strike = int(round(current_spot / strike_step) * strike_step)
    
    direction = "BULLISH" if predicted_alpha > 0 else "BEARISH"
    option_contract = f"{atm_strike} {'CE' if direction == 'BULLISH' else 'PE'}"
    
    # ATR Price Target Mapping for 20%+ Options Premium Shifts
    # Targeting a 1.5x ATR structural move ensures a high delta velocity execution
    atr_multiple_target = current_atr * 1.5
    atr_multiple_stop = current_atr * 0.75  # 1:2 Risk-to-Reward structural framing
    
    if direction == "BULLISH":
        entry_threshold = current_spot
        exit_target = current_spot + atr_multiple_target
        stop_loss = current_spot - atr_multiple_stop
    else:
        entry_threshold = current_spot
        exit_target = current_spot - atr_multiple_target
        stop_loss = current_spot + atr_multiple_stop

    return {
        "ticker": ticker.replace(".NS", ""),
        "spot": round(current_spot, 2),
        "direction": direction,
        "contract": option_contract,
        "entry": round(entry_threshold, 2),
        "target": round(exit_target, 2),
        "stop_loss": round(stop_loss, 2),
        "confidence": round(directional_accuracy * 100, 2),
        "rsi": round(float(processed_df['rsi_14'].iloc[-1]), 2),
        "atr": round(current_atr, 2)
    }

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    portfolio_matrix = []
    
    for ticker in INDIAN_TICKERS:
        try:
            metrics = execute_alpha_pipeline(ticker, f"data/{ticker}_data.csv")
            if metrics:
                portfolio_matrix.append(metrics)
        except Exception as err:
            logger.error(f"Execution boundary fault across {ticker}: {str(err)}")
            
    # Order matrix presentation sorted cleanly by directional machine model validation rank
    portfolio_matrix = sorted(portfolio_matrix, key=lambda x: x['confidence'], reverse=True)
    
    payload = {
        "meta": {
            "last_updated_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            "engine_run_status": "OPTIMIZED_STABLE"
        },
        "portfolio": portfolio_matrix
    }
    
    with open("data/predictions.json", 'w') as out_file:
        json.dump(payload, out_file, indent=4)
    logger.info("Advanced Quant Matrix written out to disk payload successfully.")

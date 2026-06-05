import os
import json
import logging
import math
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DerivativesAlphaEngine")

# Include Bank Nifty along with highly liquid F&O stocks
TICKERS = ["^NSEBANK", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS", "TCS.NS"]

class BlackScholesEngine:
    """Calculates theoretical option premiums, targets, and stops using continuous probability distributions."""
    @staticmethod
    def calculate_premium(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, sigma: float, option_type: str) -> float:
        if sigma <= 0 or time_to_expiry <= 0:
            return max(0.0, spot - strike if option_type == "CE" else strike - spot)
        
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) / (sigma * math.sqrt(time_to_expiry))
        d2 = d1 - sigma * math.sqrt(time_to_expiry)
        
        if option_type == "CE":
            premium = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            premium = strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            
        return max(1.0, float(premium))


class AdvancedFeatureEngineer:
    """Constructs non-linear, stationary features for classification training."""
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Microstructure Volatility Boundaries
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        # Oscillators & Moving Overlays
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
        
        # Classification Target: 1 if cumulative 3-bar forward return is positive, else 0
        df['forward_return'] = np.log(df['Close'].shift(-3) / df['Close'])
        df['target'] = (df['forward_return'] > 0).astype(int)
        
        return df.ffill().dropna().reset_index(drop=True)


def execute_matrix(ticker: str) -> dict:
    logger.info(f"Analyzing time-series matrices for {ticker}...")
    df = yf.download(tickers=ticker, period="30d", interval="5m", progress=False)
    if df.empty or len(df) < 500:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    processed_df = AdvancedFeatureEngineer.build(df)
    feature_cols = ['log_return', 'atr_14', 'rsi_14']
    
    X = processed_df[feature_cols]
    y = processed_df['target']
    
    # Chronological Split
    split = int(len(processed_df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train Classification Ensemble with High Regularization
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.03, l2_regularization=10.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Prediction Probability (Confidence Metric)
    latest_vector = X.iloc[[-1]]
    prob_up = float(model.predict_proba(latest_vector)[0][1])
    
    # Threshold Verification Layer (Requires strong directional conviction)
    if 0.45 < prob_up < 0.55:
        logger.info(f"Skipping {ticker}: Insufficient mathematical directional confidence.")
        return None
        
    direction = "BULLISH" if prob_up >= 0.55 else "BEARISH"
    confidence = prob_up if direction == "BULLISH" else (1 - prob_up)
    
    current_spot = float(processed_df['Close'].iloc[-1])
    
    # Asset Lot Sizes and Strike Width Rules
    if ticker == "^NSEBANK":
        strike_step = 100
        lot_size = 15
        display_name = "BANKNIFTY"
    else:
        strike_step = 50 if current_spot > 1000 else 10
        lot_size = 25 if "BANK" in ticker else 15  # Approximated fallback for stock filters
        display_name = ticker.replace(".NS", "")
        
    atm_strike = int(round(current_spot / strike_step) * strike_step)
    option_type = "CE" if direction == "BULLISH" else "PE"
    
    # Derive Option Pricing Inputs
    # Annualized historical standard deviation of log returns used as historical proxy for IV
    historical_iv = float(processed_df['log_return'].tail(375).std() * math.sqrt(252 * 75))
    time_to_expiry = 3 / 365.0  # Assume an average of 3 days remaining to weekly contracts
    risk_free_rate = 0.065     # Standard Indian RBI Repo Rate Repo benchmark
    
    theoretical_premium = BlackScholesEngine.calculate_premium(
        spot=current_spot,
        strike=atm_strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        sigma=historical_iv,
        option_type=option_type
    )
    
    # Derivative Execution Levels
    buy_entry_max = theoretical_premium
    target_exit = buy_entry_max * 1.25  # Mathematically targeted for 25% premium growth
    stop_loss = buy_entry_max * 0.85    # 15% Max risk parameter per option trade
    
    return {
        "ticker": display_name,
        "spot": round(current_spot, 2),
        "direction": direction,
        "contract": f"{atm_strike} {option_type}",
        "option_entry": round(buy_entry_max, 1),
        "option_target": round(target_exit, 1),
        "option_stop": round(stop_loss, 1),
        "lot_size": lot_size,
        "capital_per_lot": round(buy_entry_max * lot_size, 2),
        "confidence": round(confidence * 100, 1)
    }

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    active_signals = []
    
    for t in TICKERS:
        try:
            res = execute_matrix(t)
            if res:
                active_signals.append(res)
        except Exception as e:
            logger.error(f"Execution error on {t}: {str(e)}")
            
    active_signals = sorted(active_signals, key=lambda x: x['confidence'], reverse=True)
    
    payload = {
        "meta": {"timestamp_ist": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')},
        "signals": active_signals
    }
    
    with open("data/predictions.json", 'w') as f:
        json.dump(payload, f, indent=4)
    logger.info("Derivative payload successfully generated.")

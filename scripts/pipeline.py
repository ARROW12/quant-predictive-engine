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

TICKERS = [
    "^NSEBANK", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", 
    "INFY.NS", "TCS.NS", "BHARTIARTL.NS", "ITC.NS", "TATAMOTORS.NS", "MARUTI.NS"
]

class BlackScholesEngine:
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
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Base Log Returns & VIX
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vix_level'] = df['vix_close']
        df['vix_change'] = df['vix_close'].pct_change()
        
        # 2. Volatility & Momentum (ATR & RSI)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
        
        # 3. ADVANCED STRUCTURAL FEATURES (Added for max accuracy)
        
        # VWAP (Volume Weighted Average Price) - Institutional baseline
        # Resets daily to track true intraday cost basis
        df['date'] = df['Datetime'].dt.date if 'Datetime' in df.columns else df.index.date
        df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
        df['cum_vol_price'] = df.groupby('date').apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-8)
        df['dist_to_vwap'] = (df['Close'] - df['vwap']) / df['vwap']  # Mean reversion signal
        
        # RVOL (Relative Volume) - Tracks anomalous volume surges
        df['rolling_vol_20'] = df['Volume'].rolling(20).mean()
        df['rvol'] = df['Volume'] / (df['rolling_vol_20'] + 1e-8)
        
        # Multi-Timeframe Trend Proxies (Using 5m bars to simulate HTF)
        df['ema_1h_proxy'] = df['Close'].ewm(span=12, adjust=False).mean() # ~1 hr
        df['ema_1d_proxy'] = df['Close'].ewm(span=75, adjust=False).mean() # ~1 day
        df['trend_alignment'] = np.where((df['Close'] > df['ema_1h_proxy']) & (df['ema_1h_proxy'] > df['ema_1d_proxy']), 1, 
                                np.where((df['Close'] < df['ema_1h_proxy']) & (df['ema_1h_proxy'] < df['ema_1d_proxy']), -1, 0))
        
        # 4. Target Generation (Forward 3-bar prediction)
        df['forward_return'] = np.log(df['Close'].shift(-3) / df['Close'])
        df['target'] = (df['forward_return'] > 0).astype(int)
        
        # Clean up
        df = df.drop(columns=['date', 'cum_vol', 'cum_vol_price', 'rolling_vol_20'])
        return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def execute_matrix(ticker: str) -> dict:
    logger.info(f"Extracting institutional footprint arrays for {ticker}...")
    
    # 5-day data is used to ensure enough volume history for RVOL and EMAs
    df_asset = yf.download(tickers=ticker, period="5d", interval="5m", progress=False)
    df_vix = yf.download(tickers="^INDIAVIX", period="5d", interval="5m", progress=False)
    
    if df_asset.empty or df_vix.empty:
        return None
        
    if isinstance(df_asset.columns, pd.MultiIndex):
        df_asset.columns = df_asset.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    df_asset = df_asset.reset_index()
    df_vix = df_vix.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df_asset.columns else 'Date'
    
    # Ensure timezone awareness matches for safe merging
    if df_asset[time_col].dt.tz is None:
        df_asset[time_col] = df_asset[time_col].dt.tz_localize('UTC')
    if df_vix[time_col].dt.tz is None:
        df_vix[time_col] = df_vix[time_col].dt.tz_localize('UTC')
        
    df_vix = df_vix[[time_col, 'Close']].rename(columns={'Close': 'vix_close'})
    
    df_asset = df_asset.sort_values(time_col)
    df_vix = df_vix.sort_values(time_col)
    
    # Asynchronous time-series merge to prevent timestamp clipping
    df = pd.merge_asof(df_asset, df_vix, on=time_col, direction='backward')
    
    processed_df = AdvancedFeatureEngineer.build(df)
    
    # The new expanded feature vector
    feature_cols = [
        'log_return', 'atr_14', 'rsi_14', 'vix_level', 'vix_change', 
        'dist_to_vwap', 'rvol', 'trend_alignment'
    ]
    
    X = processed_df[feature_cols]
    y = processed_df['target']
    
    # Purged split: leaving a small gap between train and test to prevent data leakage
    split_idx = int(len(processed_df) * 0.8)
    gap = 3 # 3 bars gap matches our forward target window
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx+gap:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx+gap:]
    
    model = HistGradientBoostingClassifier(
        max_iter=150, 
        learning_rate=0.02, 
        l2_regularization=15.0, # High regularization to prevent overfitting noise
        max_depth=5,
        random_state=42
    )
    
    if len(np.unique(y_train)) < 2:
        return None # Prevent crash if market has only gone one direction all week
        
    model.fit(X_train, y_train)
    
    latest_vector = X.iloc[[-1]]
    prob_up = float(model.predict_proba(latest_vector)[0][1])
    
    # Strict Regime Gatekeeper: Only trade if directional bias escapes the noise band
    if 0.45 < prob_up < 0.55:
        logger.info(f"[{ticker}] Filtered out. Noise band probability: {round(prob_up, 2)}")
        return None
        
    direction = "BULLISH" if prob_up >= 0.55 else "BEARISH"
    confidence = prob_up if direction == "BULLISH" else (1 - prob_up)
    
    # Context check: Don't buy a call if the stock is miles below VWAP and in a 1H downtrend
    latest_trend = float(processed_df['trend_alignment'].iloc[-1])
    if direction == "BULLISH" and latest_trend == -1:
        logger.info(f"[{ticker}] Bullish signal aborted. Conflicts with higher timeframe downtrend.")
        return None
    if direction == "BEARISH" and latest_trend == 1:
        logger.info(f"[{ticker}] Bearish signal aborted. Conflicts with higher timeframe uptrend.")
        return None
    
    current_spot = float(processed_df['Close'].iloc[-1])
    
    if ticker == "^NSEBANK":
        strike_step = 100
        lot_size = 15
        display_name = "BANKNIFTY"
    else:
        strike_step = 50 if current_spot > 1000 else 10
        lot_size = 25 if "BANK" in ticker else 15
        display_name = ticker.replace(".NS", "")
        
    atm_strike = int(round(current_spot / strike_step) * strike_step)
    option_type = "CE" if direction == "BULLISH" else "PE"
    
    historical_iv = float(processed_df['log_return'].tail(375).std() * math.sqrt(252 * 75))
    time_to_expiry = 3 / 365.0 
    risk_free_rate = 0.065     
    
    theoretical_premium = BlackScholesEngine.calculate_premium(
        spot=current_spot, strike=atm_strike, time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate, sigma=historical_iv, option_type=option_type
    )
    
    buy_entry_max = theoretical_premium
    
    return {
        "ticker": display_name,
        "spot": round(current_spot, 2),
        "direction": direction,
        "contract": f"{atm_strike} {option_type}",
        "option_entry": round(buy_entry_max, 1),
        "option_target": round(buy_entry_max * 1.25, 1),
        "option_stop": round(buy_entry_max * 0.85, 1),
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
    logger.info("Pipeline executed successfully.")

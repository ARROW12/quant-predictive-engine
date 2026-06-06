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

# Explicit mapping of index symbols and true underlying equity lot sizes
TICKERS = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "INFY.NS", "TCS.NS", "ITC.NS"]
LOT_SIZES = {
    "^NSEI": 25,
    "^NSEBANK": 15,
    "RELIANCE.NS": 250,
    "HDFCBANK.NS": 550,
    "ICICIBANK.NS": 700,
    "SBIN.NS": 1500,
    "INFY.NS": 400,
    "TCS.NS": 175,
    "ITC.NS": 1600
}

class INDmoneyCalculator:
    """Calculates comprehensive transaction friction using 2026 post-budget STT updates."""
    @staticmethod
    def calculate_execution_plan(entry_price: float, lot_size: int, target_net_profit: float = 2000.0, target_move_pct: float = 0.10) -> dict:
        sell_price = entry_price * (1 + target_move_pct)
        lots = 1
        while True:
            qty = lots * lot_size
            buy_turnover = entry_price * qty
            sell_turnover = sell_price * qty
            total_turnover = buy_turnover + sell_turnover
            
            gross_profit = sell_turnover - buy_turnover
            
            brokerage = 40.0  
            stt = sell_turnover * 0.0015  # 2026 Budget Option STT rate (0.15%)
            exchange_txn = total_turnover * 0.0003503  
            sebi_charges = total_turnover * 0.000001  
            stamp_duty = buy_turnover * 0.00003  
            gst = (brokerage + exchange_txn + sebi_charges) * 0.18
            
            total_deductions = brokerage + stt + exchange_txn + sebi_charges + stamp_duty + gst
            net_profit = gross_profit - total_deductions
            
            if net_profit >= target_net_profit:
                return {
                    "required_lots": lots,
                    "required_qty": qty,
                    "capital_required": round(buy_turnover, 2),
                    "gross_profit": round(gross_profit, 2),
                    "total_taxes_fees": round(total_deductions, 2),
                    "net_profit": round(net_profit, 2)
                }
            lots += 1
            if lots > 500: 
                break
        return {}

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
        return max(2.0, float(premium))

class AdvancedFeatureEngineer:
    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vix_level'] = df['vix_close']
        df['vix_change'] = df['vix_close'].pct_change()
        
        # Volatility & Momentum Foundations
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi_14'] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
        
        # Volatility Squeeze Profile
        df['bb_mid'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['squeeze_factor'] = (df['bb_std'] / (df['bb_mid'] + 1e-8))
        
        # Institutional Accumulation Metrics (OBV)
        df['obv'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()
        df['obv_slope'] = df['obv'].diff(5)
        
        # Intraday VWAP Tracking
        df['date'] = df['Datetime'].dt.date if 'Datetime' in df.columns else df.index.date
        df['cum_vol'] = df.groupby('date')['Volume'].cumsum()
        df['cum_vol_price'] = df.groupby('date').apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-8)
        df['dist_to_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
        
        # Trend Coherence Matrices
        df['ema_fast'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=75, adjust=False).mean()
        df['trend_alignment'] = np.where((df['Close'] > df['ema_fast']) & (df['ema_fast'] > df['ema_slow']), 1,
                                np.where((df['Close'] < df['ema_fast']) & (df['ema_fast'] < df['ema_slow']), -1, 0))
        
        # Forward Horizons Optimization (3 bars out)
        df['forward_return'] = np.log(df['Close'].shift(-3) / df['Close'])
        df['target'] = (df['forward_return'] > 0).astype(int)
        
        drop_cols = ['date', 'cum_vol', 'cum_vol_price', 'bb_mid', 'bb_std', 'obv']
        return df.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def execute_matrix(ticker: str) -> dict:
    logger.info(f"Extracting institutional footprint arrays for {ticker}...")
    
    # Crucial Fix: Fetch 60 days to give the classifier stable and robust features
    df_asset = yf.download(tickers=ticker, period="60d", interval="5m", progress=False)
    df_vix = yf.download(tickers="^INDIAVIX", period="60d", interval="5m", progress=False)
    
    if df_asset.empty or df_vix.empty:
        return None
        
    if isinstance(df_asset.columns, pd.MultiIndex):
        df_asset.columns = df_asset.columns.get_level_values(0)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.get_level_values(0)
        
    df_asset = df_asset.reset_index()
    df_vix = df_vix.reset_index()
    
    time_col = 'Datetime' if 'Datetime' in df_asset.columns else 'Date'
    
    # CRITICAL TIMEZONE CORRECTION: Flatten timestamps to eliminate merge structural dropouts
    df_asset[time_col] = pd.to_datetime(df_asset[time_col]).dt.tz_localize(None)
    df_vix[time_col] = pd.to_datetime(df_vix[time_col]).dt.tz_localize(None)
    
    df_vix = df_vix[[time_col, 'Close']].rename(columns={'Close': 'vix_close'})
    df_asset = df_asset.sort_values(time_col)
    df_vix = df_vix.sort_values(time_col)
    
    df = pd.merge_asof(df_asset, df_vix, on=time_col, direction='backward')
    processed_df = AdvancedFeatureEngineer.build(df)
    
    feature_cols = ['log_return', 'atr_14', 'rsi_14', 'vix_level', 'vix_change', 'squeeze_factor', 'obv_slope', 'dist_to_vwap', 'trend_alignment']
    X = processed_df[feature_cols]
    y = processed_df['target']
    
    split_idx = int(len(processed_df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx+3:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx+3:]
    
    if len(np.unique(y_train)) < 2:
        return None
        
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.015, l2_regularization=20.0, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    prob_up = float(model.predict_proba(X.iloc[[-1]])[0][1])
    
    # Volatility Filter Isolation Band
    if 0.46 < prob_up < 0.54:
        return None
        
    direction = "BULLISH" if prob_up >= 0.54 else "BEARISH"
    confidence = prob_up if direction == "BULLISH" else (1 - prob_up)
    
    # Directional Check Against Macro Structure
    latest_trend = float(processed_df['trend_alignment'].iloc[-1])
    if (direction == "BULLISH" and latest_trend == -1) or (direction == "BEARISH" and latest_trend == 1):
        return None
        
    current_spot = float(processed_df['Close'].iloc[-1])
    
    # Assign Display Properties and Structural Strike Step Metrics
    if ticker == "^NSEBANK":
        strike_step, display_name = 100, "BANKNIFTY"
    elif ticker == "^NSEI":
        strike_step, display_name = 50, "NIFTY"
    else:
        strike_step = 100 if current_spot > 2000 else (50 if current_spot > 1000 else 10)
        display_name = ticker.replace(".NS", "")
        
    lot_size = LOT_SIZES.get(ticker, 25)
    atm_strike = int(round(current_spot / strike_step) * strike_step)
    option_type = "CE" if direction == "BULLISH" else "PE"
    
    historical_iv = float(processed_df['log_return'].tail(200).std() * math.sqrt(252 * 75))
    theoretical_premium = BlackScholesEngine.calculate_premium(
        spot=current_spot, strike=atm_strike, time_to_expiry=3/365.0,
        risk_free_rate=0.065, sigma=historical_iv, option_type=option_type
    )
    
    plan = INDmoneyCalculator.calculate_execution_plan(entry_price=theoretical_premium, lot_size=lot_size)
    if not plan:
        return None
        
    return {
        "ticker": display_name,
        "spot": round(current_spot, 2),
        "direction": direction,
        "contract": f"{atm_strike} {option_type}",
        "confidence": round(confidence * 100, 1),
        "option_entry": round(theoretical_premium, 1),
        "option_target": round(theoretical_premium * 1.10, 1),
        "option_stop": round(theoretical_premium * 0.85, 1),
        "lot_size": lot_size,
        "execution_plan": plan
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
            logger.error(f"Execution dropout on {t}: {str(e)}")
            
    payload = {
        "meta": {"timestamp_ist": datetime.now().strftime('%Y-%m-%d %H:%M IST')},
        "signals": sorted(active_signals, key=lambda x: x['confidence'], reverse=True)
    }
    with open("data/predictions.json", 'w') as f:
        json.dump(payload, f, indent=4)

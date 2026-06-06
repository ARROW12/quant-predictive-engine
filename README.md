# Quant Predictive Engine 📊

An automated, serverless derivatives predictive framework that orchestrates a machine learning classification pipeline to forecast short-term directional movements and compute theoretical options premiums for Bank Nifty (`^NSEBANK`) and highly liquid Indian F&O equities.

🚀 **Live Dashboard:** [arrow12.github.io/quant-predictive-engine/](https://arrow12.github.io/quant-predictive-engine/)

---

## 🏛️ System Architecture

The engine functions as an end-to-end automated quantitative pipeline running entirely via serverless infrastructure on GitHub Actions:

```text
[GitHub Actions Runner]
       │
       ├──► 1. Pulls 5m Intraday Data (yfinance) for Target Assets & ^INDIAVIX
       ├──► 2. Synchronizes time-series via As-Of Temporal Merging
       ├──► 3. Structural Features Engineered (VWAP, RVOL, ATR, RSI, VIX Momentum)
       ├──► 4. Cross-Timeframe Alignment (Simulated 1H & Daily Trend Filtering)
       ├──► 5. Purged ML Inference evaluates Directional Probability Thresholds
       ├──► 6. Evaluates Option Fair Value via Black-Scholes Mathematical Pricing
       └──► 7. Generates JSON Payload ──► Committed to Branch ──► Hosted on Pages UI
```

### Key Technical Pillars

* **Institutional Volume Profiling:** Integrates **Volume Weighted Average Price (VWAP)** and **Relative Volume (RVOL)**. The model measures the asset's distance to its true intraday cost basis and detects hidden institutional accumulation/distribution via volume surges.
* **Cross-Timeframe Context Engine:** Simulates 1-hour and daily exponential moving averages directly within the 5-minute data stream. If a 5-minute bullish algorithmic signal conflicts with a major higher-timeframe downtrend, the engine automatically aborts the trade to avoid bull traps.
* **Purged Probabilistic Classifier:** Utilizes a highly regularized `HistGradientBoostingClassifier` trained with a **Purged Data Split**. This introduces a deliberate gap between training and testing data, completely eliminating lookahead bias and time-series data leakage.
* **Capital Protection Layer:** Features a strict internal mathematical filter threshold. If the directional confidence score falls within the 45% to 55% noise band, the asset is automatically dropped from the execution grid to safeguard trading capital during choppy regimes.
* **Macro Volatility Coupling:** Uses `pd.merge_asof` to asynchronously map real-time **India VIX (`^INDIAVIX`)** volatility matrices onto stock price action vectors, capturing systemic market fear and compression regimes.
* **Continuous Option Derivative Pricing:** Processes the theoretical Black-Scholes pricing matrix to derive instant option contracts execution values:
  
  $d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})t}{\sigma\sqrt{t}}$
  
  $d_2 = d_1 - \sigma\sqrt{t}$
  
  This acts as the mathematical foundational engine calculating specific entry targets, standard lot sizes capital outlay configurations, +25% profit targets, and -15% structural stop-losses.

---

## 📂 Repository Layout

```text
├── .github/workflows/
│   └── pipeline.yml       # Serverless CRON scheduler (Runs every 5 mins during NSE hours)
├── data/
│   └── predictions.json   # Output storage matrix parsed by the web console
├── scripts/
│   └── pipeline.py        # Quantitative core data engineering & ML model script
├── index.html             # High-Conviction Derivatives Terminal UI dashboard
└── requirements.txt       # Core math, data engineering, and ML dependencies
```

---

## 🛠️ Automated Execution Schedule

The engine runs entirely in the cloud on a 5-minute interval structure aligned directly with the **National Stock Exchange (NSE)** operational trading hours (converted to UTC execution windows):

1.  **Market Open Segment:** `09:15 AM - 09:25 AM IST` (Runs every 5 minutes to capture initial opening expansion)
2.  **Core Continuous Trading:** `09:30 AM - 03:25 PM IST` (Polled every 5 minutes for trend continuation/reversals)
3.  **Market Close Segment:** `03:30 PM IST` (Final computational snapshot)

---

## 🚀 Local Installation & Deployment

To run the algorithmic discovery matrix on your local workstation:

### 1. Clone the Workspace
```bash
git clone [https://github.com/arrow12/quant-predictive-engine.git](https://github.com/arrow12/quant-predictive-engine.git)
cd quant-predictive-engine
```

### 2. Configure Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execute Local Pipelines
To run the automated data ingestion, training, and pricing steps manually:
```bash
python scripts/pipeline.py
```
This generates an updated `data/predictions.json` file on your local directory instantly.

---

## 🧪 Model Parameters & Configuration

* **Risk-Free Rate (r):** Locked at `6.5%` matching the structural benchmark RBI Repo Rate.
* **Implied Volatility (σ):** Calculated dynamically using the annualized rolling standard deviation of the asset's log returns.
* **Time to Expiry (t):** Modeled around an average weekly options execution cycle (3/365).
* **Targeting Logic:** Buy Limit ≤ Theoretical Fair Value; Profit Booking Target = +25% Premium Gain; Protection Risk Cut = -15% Premium Drop.

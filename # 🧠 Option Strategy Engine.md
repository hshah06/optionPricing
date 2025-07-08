# 🧠 Option Strategy Engine

This project is a Python-based options trading tool that:

- Uses **Black-Scholes** and **Monte Carlo simulations**
- Analyzes recent market behavior to suggest **Call or Put**
- Integrates real options chain data (via `yfinance`)
- Visualizes payoffs and highlights breakeven
- Includes metrics like **Sharpe Ratio** and **volatility analysis**

## 🔧 Features
- Custom strike price logic
- Signal-based directional bias (bullish vs bearish)
- Two-mode payoff visualization
- Real-time market data via `yfinance`

## 🛠️ Requirements

Install dependencies using:

```bash
pip install yfinance numpy pandas matplotlib scipy pylance
------- Example of Output ------- 
Current Price of TSLA: $297.81
Recent Average Daily Return: -0.452%
Annualized Volatility: 76.48%

--- Signal Analysis ---
Recent average daily return: -0.004518 (threshold: 0.001)
✓ Bearish signal: Negative recent momentum
Price position in 20-day range: 0.21 (0=low, 1=high)
✓ Bullish signal: Price near recent lows
/Users/hiravshah/optionPricing/black_scholes_engine.py:104: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  recent_price_change = (S - hist['Close'][-5]) / hist['Close'][-5]
5-day price change: -0.96%
○ Neutral: Minor recent price change

--- Option Strategy Recommendation ---
Recommended Option: BUY Put
Strike Price: $282.92
Expiration: 30 Days
BS Put Price: $18.07
MC Put Price: $18.16
Bullish Signals: 1, Bearish Signals: 1
Real Market Put Price (Strike $282.5): $1.50


-- Made By Hirav Shah --
Exploring quant finance, data science, and AI-powered strategies
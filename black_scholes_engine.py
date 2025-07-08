import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

# Black-Scholes Pricing
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Monte Carlo Simulation for Option Pricing
def monte_carlo(S, K, T, r, sigma, option_type='call', simulations=10000):
    np.random.seed(42)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(simulations))
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

# Sharpe Ratio
def calculate_sharpe(daily_returns, risk_free_rate=0.01):
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.mean(excess_returns) / np.std(excess_returns)

# Calculate expected return and volatility for directional bias
def get_market_bias(daily_returns, lookback_days=30):
    recent_returns = daily_returns[-lookback_days:] if len(daily_returns) >= lookback_days else daily_returns
    avg_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)
    return avg_return, volatility

# Main Trading Engine
def run_option_engine():
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    
    if hist.empty:
        print("Invalid ticker or no historical data available.")
        return
    
    S = hist['Close'][-1]
    daily_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    r = 0.05  # Risk-free rate
    T = 30 / 365  # 30 days to expiration
    
    # Get market bias
    avg_return, recent_vol = get_market_bias(daily_returns)
    
    # Calculate strike prices for both calls and puts
    call_strike = round(S * 1.05, 2)  # 5% OTM call
    put_strike = round(S * 0.95, 2)   # 5% OTM put
    
    # Calculate option prices
    call_bs = black_scholes(S, call_strike, T, r, sigma, 'call')
    put_bs = black_scholes(S, put_strike, T, r, sigma, 'put')
    call_mc = monte_carlo(S, call_strike, T, r, sigma, 'call')
    put_mc = monte_carlo(S, put_strike, T, r, sigma, 'put')
    
    # Improved recommendation logic based on multiple factors
    print(f"\nCurrent Price of {ticker}: ${S:.2f}")
    print(f"Recent Average Daily Return: {avg_return*100:.3f}%")
    print(f"Annualized Volatility: {sigma*100:.2f}%")
    
    # Decision factors
    bullish_signals = 0
    bearish_signals = 0
    
    print(f"\n--- Signal Analysis ---")
    
    # Factor 1: Recent price momentum
    print(f"Recent average daily return: {avg_return:.6f} (threshold: 0.001)")
    if avg_return > 0.0001:  # Lowered threshold to be less restrictive
        bullish_signals += 1
        print("✓ Bullish signal: Positive recent momentum")
    else:
        bearish_signals += 1
        print("✓ Bearish signal: Negative recent momentum")
    
    # Factor 2: Price relative to recent range
    recent_high = hist['High'][-20:].max()
    recent_low = hist['Low'][-20:].min()
    price_position = (S - recent_low) / (recent_high - recent_low)
    
    print(f"Price position in 20-day range: {price_position:.2f} (0=low, 1=high)")
    if price_position > 0.75:  # Made slightly more restrictive
        bearish_signals += 1
        print("✓ Bearish signal: Price near recent highs")
    elif price_position < 0.25:  # Made slightly more restrictive
        bullish_signals += 1
        print("✓ Bullish signal: Price near recent lows")
    else:
        print("○ Neutral: Price in middle of recent range")
    
    # Factor 3: Recent price change
    recent_price_change = (S - hist['Close'][-5]) / hist['Close'][-5]
    print(f"5-day price change: {recent_price_change*100:.2f}%")
    if recent_price_change > 0.02:  # 2% positive change
        bullish_signals += 1
        print("✓ Bullish signal: Strong recent price increase")
    elif recent_price_change < -0.02:  # 2% negative change
        bearish_signals += 1
        print("✓ Bearish signal: Strong recent price decrease")
    else:
        print("○ Neutral: Minor recent price change")
    
    # Factor 4: Volatility regime
    if recent_vol > sigma / np.sqrt(252):  # Recent vol higher than historical
        print("✓ High volatility environment detected")
        # In high vol, prefer direction with more conviction
        if bullish_signals > bearish_signals:
            bullish_signals += 1
            print("✓ Bonus bullish signal: High vol + bullish trend")
        elif bearish_signals > bullish_signals:
            bearish_signals += 1
            print("✓ Bonus bearish signal: High vol + bearish trend")
    
    # Make recommendation
    if bullish_signals > bearish_signals:
        recommendation = "Call"
        strike = call_strike
        bs_price = call_bs
        mc_price = call_mc
    else:
        recommendation = "Put"
        strike = put_strike
        bs_price = put_bs
        mc_price = put_mc
    
    print(f"\n--- Option Strategy Recommendation ---")
    print(f"Recommended Option: BUY {recommendation}")
    print(f"Strike Price: ${strike}")
    print(f"Expiration: 30 Days")
    print(f"BS {recommendation} Price: ${bs_price:.2f}")
    print(f"MC {recommendation} Price: ${mc_price:.2f}")
    print(f"Bullish Signals: {bullish_signals}, Bearish Signals: {bearish_signals}")
    
    # Real Options Chain (if available)
    try:
        if recommendation == "Call":
            options_df = stock.option_chain().calls
            closest_strike = options_df.iloc[(options_df['strike'] - strike).abs().argsort()[:1]]
        else:
            options_df = stock.option_chain().puts
            closest_strike = options_df.iloc[(options_df['strike'] - strike).abs().argsort()[:1]]
        
        if not closest_strike.empty:
            best = closest_strike.iloc[0]
            print(f"Real Market {recommendation} Price (Strike ${best['strike']}): ${best['lastPrice']:.2f}")
    except Exception as e:
        print("Live option chain data not available for this ticker.")
    
    # Payoff Diagram
    x = np.linspace(S * 0.7, S * 1.3, 100)
    call_payoff = np.maximum(x - call_strike, 0) - call_bs
    put_payoff = np.maximum(put_strike - x, 0) - put_bs
    
    plt.figure(figsize=(12, 8))
    
    # Plot both payoffs
    plt.subplot(2, 1, 1)
    plt.plot(x, call_payoff, label="Call Payoff", color='green', linewidth=2)
    plt.plot(x, put_payoff, label="Put Payoff", color='red', linewidth=2)
    plt.axvline(S, linestyle='--', color='gray', label='Current Price', alpha=0.7)
    plt.axhline(0, linestyle=':', color='black', alpha=0.5)
    plt.title(f"{ticker} Option Strategy Payoff Comparison")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Highlight recommended option
    plt.subplot(2, 1, 2)
    if recommendation == "Call":
        plt.plot(x, call_payoff, label=f"Recommended: {recommendation}", color='green', linewidth=3)
        plt.axvline(call_strike, linestyle='--', color='green', label=f'Strike: ${call_strike}', alpha=0.7)
    else:
        plt.plot(x, put_payoff, label=f"Recommended: {recommendation}", color='red', linewidth=3)
        plt.axvline(put_strike, linestyle='--', color='red', label=f'Strike: ${put_strike}', alpha=0.7)
    
    plt.axvline(S, linestyle='--', color='gray', label='Current Price', alpha=0.7)
    plt.axhline(0, linestyle=':', color='black', alpha=0.5)
    plt.title(f"Recommended {recommendation} Option Strategy")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Sharpe Ratio and additional metrics
    sharpe = calculate_sharpe(daily_returns)
    print(f"\nAdditional Metrics:")
    print(f"Sharpe Ratio (6mo): {sharpe:.2f}")
    print(f"Price Position in 20-day range: {price_position:.2f} (0=low, 1=high)")
    
    # Breakeven analysis
    if recommendation == "Call":
        breakeven = strike + bs_price
        print(f"Call Breakeven: ${breakeven:.2f} ({((breakeven/S - 1)*100):+.1f}% from current)")
    else:
        breakeven = strike - bs_price
        print(f"Put Breakeven: ${breakeven:.2f} ({((breakeven/S - 1)*100):+.1f}% from current)")

# Run
if __name__ == "__main__":
    run_option_engine()
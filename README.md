# Trading Bot Demo

A professional Python-based trading bot demonstration showcasing algorithmic trading capabilities, quantitative analysis, and backtesting with performance visualization for global markets.

## Overview

This demo implements a simple but effective **Moving Average Crossover Strategy** using:
- **20-day Simple Moving Average (SMA)**
- **50-day Simple Moving Average (SMA)**
- **Buy Signal**: When 20-day SMA crosses above 50-day SMA
- **Sell Signal**: When 20-day SMA crosses below 50-day SMA

## Features

✅ **Universal Market Support**: Works with any stock market globally via Yahoo Finance  
✅ **Professional Backtesting**: Complete trade simulation with transaction costs  
✅ **Performance Metrics**: Total return, win rate, max drawdown, Sharpe ratio  
✅ **Professional Charts**: Price/signals chart and equity curve visualization  
✅ **Robust Error Handling**: Graceful handling of API failures and data issues  

## Requirements

**Python 3.11+ Required**

```bash
pip install -r requirements.txt
```

## Usage

```bash
python trading_bot_demo.py
```

## Output

The demo generates:
1. **Terminal Results**: Complete performance summary
2. **price_and_signals.png**: Stock price with moving averages and buy/sell signals
3. **equity_curve.png**: Portfolio performance vs buy-and-hold benchmark

## Sample Output

```
=== Trading Bot Demo Results ===
Stock: RELIANCE.NS
Period: 2023-09-20 to 2025-09-19
Strategy: Moving Average Crossover (20/50)

Initial Capital: ₹1,00,000
Final Portfolio Value: ₹1,05,489
Total Return: 5.49%

Trading Statistics:
Total Trades: 8
Winning Trades: 1
Win Rate: 25.0%
Max Drawdown: 18.67%
Sharpe Ratio: 0.29

Charts saved to: results/
```

## 🌍 Market Adaptability

This demo is designed to work with **any global stock market** with minimal changes:

### **Supported Markets:**
- **US Markets**: AAPL, MSFT, GOOGL (no suffix needed)
- **Indian Markets**: RELIANCE.NS, TCS.NS, INFY.NS 
- **UK Markets**: VODL.L, BP.L, LLOY.L
- **European Markets**: SAP.DE, ASML.AS, NESN.SW
- **100+ Global Exchanges** supported via Yahoo Finance

### **What Makes It Universal:**
- **Strategy Logic**: Moving averages work on any price data globally
- **Data Source**: Yahoo Finance API supports worldwide markets
- **Performance Metrics**: Financial calculations are market-independent
- **Backtesting Engine**: Universal trade simulation logic
- **Visualization**: Charts work with any currency/market data

### **Easy Customization:**
```python
# Change these 2 lines for any market:
TICKER = "AAPL"        # US: Apple Inc.
TICKER = "RELIANCE.NS" # India: Reliance Industries
TICKER = "VODL.L"      # UK: Vodafone Group
TICKER = "SAP.DE"      # Germany: SAP SE

# Update currency display (cosmetic only):
# $ for US, ₹ for India, £ for UK, € for Europe
```

## Technical Implementation

- **Language**: Python 3.11+
- **Data Source**: Yahoo Finance API via yfinance (global market support)
- **Strategy**: Moving Average Crossover (universal technical analysis)
- **Backtesting**: Transaction cost modeling (adaptable to any market)
- **Visualization**: Professional matplotlib charts
- **Code Quality**: Type hints, error handling, comprehensive logging

## 📸 Portfolio Demonstration

### Screenshot Instructions
1. **Run the demo**: `python trading_bot_demo.py`
2. **Capture terminal**: Screenshot the complete backtest results output
3. **Chart files**: Use generated PNG files from `results/` folder

### 🚀 Skills Demonstrated
- **Python 3.11+**: Modern Python development with type hints and error handling
- **Financial Data Analysis**: pandas, numpy, yfinance API integration
- **Quantitative Strategy**: Moving average crossover with backtesting
- **Data Visualization**: Professional matplotlib charts and analysis
- **Indian Markets**: NSE stock handling (.NS format) with ₹ currency
- **Performance Analytics**: Sharpe ratio, drawdown, win rate calculations

### 📋 Portfolio Assets
- **backtest_terminal_output.png**: Terminal screenshot showing live backtesting execution
- **price_and_signals.png**: Trading strategy visualization with buy/sell markers  
- **equity_curve.png**: Portfolio performance vs benchmark comparison
- **Source Code**: 500 lines of documented, production-quality Python

### 🎨 Portfolio Descriptions

**Short**: *"Professional Python trading bot demonstration with algorithmic backtesting for global markets"*

**Medium**: *"Professional trading bot demonstration showcasing algorithmic trading capabilities, quantitative analysis, moving average crossover strategy, and comprehensive backtesting with performance visualization for global stock markets using Python 3.11+."*

**Detailed**: *"Complete algorithmic trading system showcasing quantitative finance expertise through a universal backtesting engine. Features moving average crossover strategy, real-time data fetching via Yahoo Finance API (100+ global exchanges), transaction cost modeling, comprehensive performance analytics (Sharpe ratio, max drawdown, win rate), and professional visualization. Demonstrates Python 3.11+ development, financial data analysis, and production-quality code standards with global market adaptability."*

### 📊 Expected Demo Results
- **Return**: ~5-6% over 2-year backtest period
- **Total Trades**: 8-10 trades executed  
- **Win Rate**: 20-30% (realistic for simple strategy)
- **Max Drawdown**: 15-20%
- **Charts**: Two professional PNG visualizations generated

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is a demonstration project for educational and portfolio purposes only. Not intended for actual trading or investment advice.

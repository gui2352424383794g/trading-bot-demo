#!/usr/bin/env python3
"""
Trading Bot Demo - Professional Portfolio Showcase
A minimal but professional trading bot demonstration showcasing algorithmic trading 
capabilities, quantitative analysis, and Python development expertise.

Author: Gaurav Kumar Gupta
Version: 1.0
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TICKER = "RELIANCE.NS"  # Reliance Industries Limited - most liquid Indian stock
PERIOD = "2y"
INITIAL_CAPITAL = 100000.0  # ₹1,00,000
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade
SHORT_MA_PERIOD = 20
LONG_MA_PERIOD = 50
RESULTS_DIR = "results"


def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance with robust error handling.
    
    Args:
        ticker: Stock symbol (e.g., 'RELIANCE.NS')
        period: Time period (e.g., '2y')
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If no data is available or data quality is poor
        ConnectionError: If API connection fails
    """
    logger.info(f"Fetching stock data for {ticker} (period: {period})")
    
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns by taking the first level (OHLCV names)
            data.columns = [col[0] for col in data.columns]
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data quality
        if len(data) < 100:
            raise ValueError(f"Insufficient data: only {len(data)} records (minimum 100 required)")
        
        # Check for excessive null values
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if null_percentage > 10:
            raise ValueError(f"Poor data quality: {null_percentage:.1f}% null values")
        
        # Fill any remaining null values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Successfully fetched {len(data)} records for {ticker}")
        logger.info(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        raise ConnectionError(f"Data fetch failed: {str(e)}")


def calculate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals using moving average crossover strategy.
    
    Strategy: Buy when 20-day SMA crosses above 50-day SMA
             Sell when 20-day SMA crosses below 50-day SMA
    
    Args:
        data: Stock price data with OHLCV columns
        
    Returns:
        DataFrame with additional signal columns
    """
    logger.info("Calculating moving average crossover signals")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Calculate Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=SHORT_MA_PERIOD).mean()
    df['SMA_50'] = df['Close'].rolling(window=LONG_MA_PERIOD).mean()
    
    # Initialize signal columns
    df['Signal'] = 0  # 0 = Hold, 1 = Buy, -1 = Sell
    df['Position'] = 0  # Track current position
    
    # Generate signals based on MA crossover
    for i in range(1, len(df)):
        if pd.notna(df['SMA_20'].iloc[i]) and pd.notna(df['SMA_50'].iloc[i]):
            # Buy signal: 20-day SMA crosses above 50-day SMA
            if (df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] and 
                df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = 1
                
            # Sell signal: 20-day SMA crosses below 50-day SMA
            elif (df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] and 
                  df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = -1
    
    # Track position (1 = Long, 0 = No position)
    position = 0
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1:  # Buy signal
            position = 1
        elif df['Signal'].iloc[i] == -1:  # Sell signal
            position = 0
        df.loc[df.index[i], 'Position'] = position
    
    # Count signals
    buy_signals = len(df[df['Signal'] == 1])
    sell_signals = len(df[df['Signal'] == -1])
    
    logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    return df


def run_backtest(data: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    """
    Execute simple backtest simulation with transaction costs.
    
    Args:
        data: DataFrame with price data and signals
        initial_capital: Starting capital in rupees
        
    Returns:
        Dictionary containing portfolio values, trades, and metrics
    """
    logger.info(f"Running backtest with initial capital: ₹{initial_capital:,.2f}")
    
    # Initialize tracking variables
    capital = initial_capital
    shares = 0
    portfolio_values = []
    trades = []
    
    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        signal = data['Signal'].iloc[i]
        date = data.index[i]
        
        # Calculate current portfolio value
        portfolio_value = capital + (shares * current_price)
        portfolio_values.append(portfolio_value)
        
        # Execute trades based on signals
        if signal == 1 and shares == 0:  # Buy signal and not already holding
            # Calculate shares to buy (invest all available capital)
            transaction_cost = capital * TRANSACTION_COST_PCT
            available_capital = capital - transaction_cost
            shares_to_buy = int(available_capital / current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                shares = shares_to_buy
                capital = capital - cost - transaction_cost
                
                trade = {
                    'date': date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'cost': cost,
                    'transaction_cost': transaction_cost,
                    'capital_remaining': capital
                }
                trades.append(trade)
                logger.debug(f"BUY: {shares_to_buy} shares at ₹{current_price:.2f}")
        
        elif signal == -1 and shares > 0:  # Sell signal and currently holding
            # Sell all shares
            proceeds = shares * current_price
            transaction_cost = proceeds * TRANSACTION_COST_PCT
            net_proceeds = proceeds - transaction_cost
            
            trade = {
                'date': date,
                'type': 'SELL',
                'price': current_price,
                'shares': shares,
                'proceeds': proceeds,
                'transaction_cost': transaction_cost,
                'net_proceeds': net_proceeds
            }
            
            # Calculate P&L for this trade pair
            if trades:  # Find the corresponding buy trade
                last_buy = next((t for t in reversed(trades) if t['type'] == 'BUY'), None)
                if last_buy:
                    pnl = net_proceeds - (last_buy['cost'] + last_buy['transaction_cost'])
                    trade['pnl'] = pnl
            
            capital += net_proceeds
            shares = 0
            trades.append(trade)
            logger.debug(f"SELL: {trade['shares']} shares at ₹{current_price:.2f}")
    
    # Handle final portfolio value if still holding shares
    if shares > 0:
        final_value = capital + (shares * data['Close'].iloc[-1])
    else:
        final_value = capital
    
    # Create portfolio values series
    portfolio_series = pd.Series(portfolio_values, index=data.index)
    
    logger.info(f"Backtest completed: {len(trades)} trades executed")
    logger.info(f"Final portfolio value: ₹{final_value:,.2f}")
    
    return {
        'portfolio_values': portfolio_series,
        'trades': trades,
        'final_value': final_value,
        'final_capital': capital,
        'final_shares': shares
    }


def calculate_metrics(portfolio_values: pd.Series, trades: List[Dict], 
                     initial_capital: float) -> Dict[str, Any]:
    """
    Calculate essential performance metrics.
    
    Args:
        portfolio_values: Series of daily portfolio values
        trades: List of executed trades
        initial_capital: Starting capital
        
    Returns:
        Dictionary of performance metrics
    """
    logger.info("Calculating performance metrics")
    
    # Basic metrics
    final_value = portfolio_values.iloc[-1]
    total_return = final_value - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    # Trade statistics
    total_trades = len([t for t in trades if 'pnl' in t])
    if total_trades > 0:
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * 100
    else:
        winning_trades = 0
        win_rate = 0
    
    # Calculate daily returns for risk metrics
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Maximum drawdown calculation
    cumulative_returns = (portfolio_values / initial_capital)
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    # Sharpe ratio (simplified - assuming 0% risk-free rate)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'completed_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio
    }
    
    logger.info(f"Metrics calculated - Return: {total_return_pct:.2f}%, Win Rate: {win_rate:.1f}%")
    
    return metrics


def create_charts(data: pd.DataFrame, portfolio_values: pd.Series, 
                  metrics: Dict[str, Any]) -> None:
    """
    Generate and save professional trading charts.
    
    Args:
        data: Stock data with signals
        portfolio_values: Portfolio value over time
        metrics: Performance metrics
    """
    logger.info("Creating professional charts")
    
    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Set up matplotlib style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Chart 1: Price and Signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Plot price and moving averages
    ax1.plot(data.index, data['Close'], label='RELIANCE Price', color='blue', linewidth=1.5)
    ax1.plot(data.index, data['SMA_20'], label='20-day SMA', color='orange', linewidth=1)
    ax1.plot(data.index, data['SMA_50'], label='50-day SMA', color='red', linewidth=1)
    
    # Plot buy/sell signals
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', 
               marker='^', s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', 
               marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title('RELIANCE.NS - Moving Average Crossover Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (₹)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Volume subplot
    ax2.bar(data.index, data['Volume'], color='gray', alpha=0.6, width=1)
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/price_and_signals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Equity Curve
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot portfolio value
    ax.plot(portfolio_values.index, portfolio_values, label='Portfolio Value', 
           color='blue', linewidth=2)
    
    # Calculate and plot buy-and-hold benchmark
    buy_hold_values = (data['Close'] / data['Close'].iloc[0]) * INITIAL_CAPITAL
    ax.plot(buy_hold_values.index, buy_hold_values, label='Buy & Hold', 
           color='gray', linestyle='--', linewidth=1.5)
    
    # Add performance text box
    textstr = f"""Performance Summary:
Total Return: {metrics['total_return_pct']:.2f}%
Win Rate: {metrics['win_rate']:.1f}%
Max Drawdown: {metrics['max_drawdown_pct']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Total Trades: {metrics['total_trades']}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    ax.set_title('Portfolio Performance - Equity Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Format y-axis with rupee formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/equity_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Charts saved to {RESULTS_DIR}/")


def display_results(metrics: Dict[str, Any], data: pd.DataFrame) -> None:
    """
    Display formatted results to terminal.
    
    Args:
        metrics: Performance metrics dictionary
        data: Stock data for period information
    """
    print("\n" + "="*50)
    print("=== Trading Bot Demo Results ===")
    print("="*50)
    print(f"Stock: {TICKER}")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Strategy: Moving Average Crossover ({SHORT_MA_PERIOD}/{LONG_MA_PERIOD})")
    print()
    print(f"Initial Capital: ₹{metrics['initial_capital']:,.0f}")
    print(f"Final Portfolio Value: ₹{metrics['final_value']:,.0f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print()
    print("Trading Statistics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['completed_trades'] - metrics['winning_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print()
    print(f"Charts saved to: {RESULTS_DIR}/")
    print("- price_and_signals.png")
    print("- equity_curve.png")
    print()
    print("Demo completed successfully!")
    print("="*50)


def main() -> None:
    """Main execution function orchestrating the complete demo flow."""
    logger.info("Starting Trading Bot Demo...")
    
    try:
        # 1. Fetch stock data
        logger.info("Step 1: Fetching stock data")
        data = fetch_stock_data(TICKER, PERIOD)
        
        # 2. Generate trading signals
        logger.info("Step 2: Generating trading signals")
        data_with_signals = calculate_signals(data)
        
        # 3. Run backtest simulation
        logger.info("Step 3: Running backtest simulation")
        backtest_results = run_backtest(data_with_signals, INITIAL_CAPITAL)
        
        # 4. Calculate performance metrics
        logger.info("Step 4: Calculating performance metrics")
        metrics = calculate_metrics(
            backtest_results['portfolio_values'], 
            backtest_results['trades'], 
            INITIAL_CAPITAL
        )
        
        # 5. Create professional visualizations
        logger.info("Step 5: Creating visualizations")
        create_charts(data_with_signals, backtest_results['portfolio_values'], metrics)
        
        # 6. Display results
        logger.info("Step 6: Displaying results")
        display_results(metrics, data_with_signals)
        
        logger.info("Trading Bot Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please check your internet connection and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Portfolio settings
TICKERS = ["SPY", "AAPL", "QQQ", "EFA", "GOOGL"]
WEIGHTS = [0.5522, 0.1646, 0.1327, 0.0885, 0.0619]  # Equal weighting, modify as needed
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate, adjust as needed
TRADING_DAYS = 252  # Trading days in a year

def fetch_portfolio_data(tickers, period="3y"):
    """
    Fetch historical data for portfolio
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 3y, 5y, 10y, ytd, max
    """
    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, period=period, progress=False)
    
    if len(tickers) == 1:
        # Handle single ticker case
        adj_close = pd.DataFrame(data["Adj Close"])
        adj_close.columns = tickers
    else:
        adj_close = data["Close"]  # When auto_adjust=True, 'Close' column contains split/dividend adjusted data    
    return adj_close.dropna()

def calculate_portfolio_metrics(prices, weights, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate comprehensive portfolio metrics
    """
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Portfolio returns (weighted average)
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 1. Individual Stock Metrics
    individual_metrics = pd.DataFrame(index=TICKERS)
    
    # Annualized returns and volatility
    individual_metrics['Annual_Return'] = (1 + returns.mean()) ** TRADING_DAYS - 1
    individual_metrics['Annual_Volatility'] = returns.std() * np.sqrt(TRADING_DAYS)
    
    # Sharpe Ratio
    individual_metrics['Sharpe_Ratio'] = (individual_metrics['Annual_Return'] - risk_free_rate) / individual_metrics['Annual_Volatility']
    
    # Maximum Drawdown
    for ticker in TICKERS:
        cum_returns = (1 + returns[ticker]).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        individual_metrics.loc[ticker, 'Max_Drawdown'] = drawdown.min()
    
    # Beta to SPY (market)
    if 'SPY' in returns.columns:
        for ticker in TICKERS:
            if ticker != 'SPY':
                cov = returns[ticker].cov(returns['SPY'])
                var = returns['SPY'].var()
                individual_metrics.loc[ticker, 'Beta_vs_SPY'] = cov / var
    
    # 2. Portfolio Metrics
    portfolio_annual_return = (1 + portfolio_returns.mean()) ** TRADING_DAYS - 1
    portfolio_annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    portfolio_sharpe = (portfolio_annual_return - risk_free_rate) / portfolio_annual_volatility
    
    # Portfolio cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    # Portfolio drawdown
    running_max = portfolio_cumulative.expanding().max()
    portfolio_drawdown = (portfolio_cumulative - running_max) / running_max
    portfolio_max_drawdown = portfolio_drawdown.min()
    
    # 3. Correlation Matrix
    correlation_matrix = returns.corr()
    
    # 4. Covariance Matrix
    covariance_matrix = returns.cov() * TRADING_DAYS  # Annualized covariance
    
    # 5. Portfolio variance and risk
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    
    # 6. Diversification ratio
    weighted_vol = np.sum([weights[i] * individual_metrics.loc[ticker, 'Annual_Volatility'] 
                          for i, ticker in enumerate(TICKERS)])
    diversification_ratio = weighted_vol / portfolio_risk
    
    # 7. Value at Risk (VaR) - Historical method, 95% confidence
    var_95 = np.percentile(portfolio_returns, 5)
    var_95_annual = var_95 * np.sqrt(TRADING_DAYS)
    
    # Compile results
    results = {
        'individual_metrics': individual_metrics,
        'portfolio_metrics': {
            'Annual Return': portfolio_annual_return,
            'Annual Volatility': portfolio_annual_volatility,
            'Sharpe Ratio': portfolio_sharpe,
            'Max Drawdown': portfolio_max_drawdown,
            'Portfolio Risk': portfolio_risk,
            'Diversification Ratio': diversification_ratio,
            'VaR (95%, Annual)': var_95_annual
        },
        'correlation_matrix': correlation_matrix,
        'covariance_matrix': covariance_matrix,
        'returns': returns,
        'portfolio_returns': portfolio_returns,
        'prices': prices
    }
    
    return results

def visualize_portfolio_analysis(results):
    """
    Create comprehensive visualizations
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Correlation Heatmap
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(results['correlation_matrix'], annot=True, cmap='coolwarm', 
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title('Correlation Matrix')
    
    # 2. Individual Returns vs Volatility
    ax2 = plt.subplot(3, 3, 2)
    metrics = results['individual_metrics']
    ax2.scatter(metrics['Annual_Volatility'], metrics['Annual_Return'], 
                s=100, alpha=0.7)
    
    # Annotate points
    for ticker in TICKERS:
        ax2.annotate(ticker, (metrics.loc[ticker, 'Annual_Volatility'], 
                             metrics.loc[ticker, 'Annual_Return']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Volatility')
    ax2.set_ylabel('Return')
    ax2.set_title('Risk-Return Profile')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sharpe Ratios
    ax3 = plt.subplot(3, 3, 3)
    sharpe_ratios = metrics['Sharpe_Ratio'].sort_values()
    colors = ['green' if x > 0 else 'red' for x in sharpe_ratios]
    sharpe_ratios.plot(kind='barh', color=colors, ax=ax3)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Sharpe Ratios')
    ax3.set_xlabel('Sharpe Ratio')
    
    # 4. Cumulative Returns Comparison
    ax4 = plt.subplot(3, 3, 4)
    normalized_prices = results['prices'] / results['prices'].iloc[0]
    normalized_prices.plot(ax=ax4)
    ax4.set_title('Normalized Price Performance')
    ax4.set_ylabel('Growth of $1')
    ax4.legend(loc='upper left', fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    # 5. Portfolio Cumulative Returns
    ax5 = plt.subplot(3, 3, 5)
    portfolio_cumulative = (1 + results['portfolio_returns']).cumprod()
    portfolio_cumulative.plot(ax=ax5, color='purple', linewidth=2)
    ax5.set_title('Portfolio Cumulative Returns')
    ax5.set_ylabel('Growth of $1')
    ax5.grid(True, alpha=0.3)
    
    # 6. Drawdown Chart
    ax6 = plt.subplot(3, 3, 6)
    running_max = portfolio_cumulative.expanding().max()
    drawdown = (portfolio_cumulative - running_max) / running_max
    drawdown.plot(ax=ax6, color='red', alpha=0.7)
    ax6.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax6.set_title('Portfolio Drawdown')
    ax6.set_ylabel('Drawdown %')
    ax6.grid(True, alpha=0.3)
    
    # 7. Rolling Correlation (example: AAPL vs SPY)
    ax7 = plt.subplot(3, 3, 7)
    if 'AAPL' in results['returns'].columns and 'SPY' in results['returns'].columns:
        rolling_corr = results['returns']['AAPL'].rolling(window=60).corr(results['returns']['SPY'])
        rolling_corr.plot(ax=ax7, color='blue')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.set_title('60-Day Rolling Correlation: AAPL vs SPY')
        ax7.set_ylabel('Correlation')
        ax7.grid(True, alpha=0.3)
    
    # 8. Returns Distribution
    ax8 = plt.subplot(3, 3, 8)
    results['portfolio_returns'].hist(bins=50, alpha=0.7, edgecolor='black', ax=ax8)
    ax8.axvline(x=results['portfolio_returns'].mean(), color='red', 
                linestyle='--', label=f'Mean: {results["portfolio_returns"].mean():.4f}')
    ax8.set_title('Portfolio Daily Returns Distribution')
    ax8.set_xlabel('Daily Returns')
    ax8.set_ylabel('Frequency')
    ax8.legend()
    
    # 9. Portfolio Metrics Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    portfolio_text = "PORTFOLIO METRICS\n\n"
    for metric, value in results['portfolio_metrics'].items():
        if 'Return' in metric or 'Ratio' in metric:
            portfolio_text += f"{metric}: {value:.3f}\n"
        elif 'Drawdown' in metric:
            portfolio_text += f"{metric}: {value:.3%}\n"
        elif 'Risk' in metric or 'Volatility' in metric:
            portfolio_text += f"{metric}: {value:.3%}\n"
        else:
            portfolio_text += f"{metric}: {value:.3f}\n"
    
    ax9.text(0.1, 0.5, portfolio_text, fontsize=10, 
             verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('Portfolio Analysis: SPY, AAPL, QQQ, EFA, GOOGL', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def print_detailed_report(results):
    """
    Print detailed text report
    """
    print("=" * 70)
    print("PORTFOLIO ANALYSIS REPORT")
    print("=" * 70)
    
    print("\n1. INDIVIDUAL STOCK METRICS:")
    print("-" * 70)
    metrics = results['individual_metrics']
    print(metrics.round(4))
    
    print("\n2. PORTFOLIO METRICS:")
    print("-" * 70)
    for metric, value in results['portfolio_metrics'].items():
        if 'Return' in metric or 'Ratio' in metric:
            print(f"{metric:<25}: {value:.4f}")
        elif 'Drawdown' in metric:
            print(f"{metric:<25}: {value:.2%}")
        elif 'Risk' in metric or 'Volatility' in metric:
            print(f"{metric:<25}: {value:.2%}")
        else:
            print(f"{metric:<25}: {value:.4f}")
    
    print("\n3. CORRELATION MATRIX:")
    print("-" * 70)
    print(results['correlation_matrix'].round(3))
    
    print("\n4. COVARIANCE MATRIX (Annualized):")
    print("-" * 70)
    print(results['covariance_matrix'].round(6))
    
    print("\n5. KEY OBSERVATIONS:")
    print("-" * 70)
    
    # Identify highest correlations
    corr_matrix = results['correlation_matrix']
    high_corr_pairs = []
    for i in range(len(TICKERS)):
        for j in range(i+1, len(TICKERS)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((TICKERS[i], TICKERS[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print("High Correlation Pairs (>0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No extremely high correlations (>0.7) found.")
    
    # Diversification effectiveness
    div_ratio = results['portfolio_metrics']['Diversification Ratio']
    if div_ratio > 1.5:
        print(f"\n✓ Good diversification (Ratio: {div_ratio:.2f})")
    else:
        print(f"\n⚠ Moderate diversification (Ratio: {div_ratio:.2f})")
    
    # Sharpe ratio assessment
    sharpe = results['portfolio_metrics']['Sharpe Ratio']
    if sharpe > 1:
        print(f"✓ Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
    elif sharpe > 0:
        print(f"✓ Positive risk-adjusted returns (Sharpe: {sharpe:.2f})")
    else:
        print(f"⚠ Negative risk-adjusted returns (Sharpe: {sharpe:.2f})")

# Main execution
if __name__ == "__main__":
    # Fetch data
    prices = fetch_portfolio_data(TICKERS, period="3y")
    
    # Calculate metrics
    results = calculate_portfolio_metrics(prices, WEIGHTS)
    
    # Print report
    print_detailed_report(results)
    
    # Visualize
    visualize_portfolio_analysis(results)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize

st.set_page_config(page_title="Optimal Portfolio Analysis", layout="wide")
st.title("Optimal Portfolio Analysis Dashboard")

def get_data():
    tickers = ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'CRM']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return tickers, returns

def portfolio_stats(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = portfolio_return / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe

def optimize_portfolio(returns):
    def neg_sharpe(weights):
        return -portfolio_stats(weights, returns)[2]
    
    n_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x

def generate_efficient_frontier(returns, n_portfolios=1000):
    n_assets = len(returns.columns)
    returns_array = np.zeros(n_portfolios)
    volatility_array = np.zeros(n_portfolios)
    sharpe_array = np.zeros(n_portfolios)
    
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights = weights/np.sum(weights)
        ret, vol, sharpe = portfolio_stats(weights, returns)
        returns_array[i] = ret
        volatility_array[i] = vol
        sharpe_array[i] = sharpe
    
    return returns_array, volatility_array, sharpe_array

# Get data
tickers, returns = get_data()

# Optimize portfolio
optimal_weights = optimize_portfolio(returns)
opt_return, opt_vol, opt_sharpe = portfolio_stats(optimal_weights, returns)

# Display optimal portfolio
st.header("Optimal Portfolio Weights")
weights_df = pd.DataFrame({
    'Stock': tickers,
    'Weight': optimal_weights
})
st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}))

# Display metrics
st.header("Portfolio Metrics")
metrics_df = pd.DataFrame({
    'Metric': ['Expected Return', 'Expected Volatility', 'Sharpe Ratio'],
    'Value': [f"{opt_return:.2%}", f"{opt_vol:.2%}", f"{opt_sharpe:.2f}"]
})
st.dataframe(metrics_df)

# Generate and plot efficient frontier
returns_array, volatility_array, sharpe_array = generate_efficient_frontier(returns)

st.header("Efficient Frontier")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(volatility_array, returns_array, 
                    c=sharpe_array, cmap='viridis', 
                    marker='o', s=10)
ax.scatter(opt_vol, opt_return, 
          color='red', marker='*', s=200, 
          label='Optimal Portfolio')
ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
plt.colorbar(scatter, label='Sharpe Ratio')
ax.legend()
st.pyplot(fig)

# Correlation matrix
st.header("Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)

# Historical performance
st.header("Historical Performance")
fig, ax = plt.subplots(figsize=(12, 6))
portfolio_returns = returns.dot(optimal_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()
cumulative_returns.plot(ax=ax)
ax.set_title('Optimal Portfolio Performance')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.grid(True)
st.pyplot(fig)

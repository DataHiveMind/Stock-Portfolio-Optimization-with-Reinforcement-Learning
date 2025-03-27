# src/evaluate_results.py
import pandas as pd
import numpy as np

def evaluate_results(csv_path):
    """Evaluates backtesting results."""
    df = pd.read_csv(csv_path, index_col=0)
    returns = df['Balance'].pct_change().dropna()
    total_return = (df['Balance'].iloc[-1] / df['Balance'].iloc[0]) - 1
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) # Assuming daily data
    max_drawdown = (returns.cumsum().cummax() - returns.cumsum()).max()

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")

if __name__ == "__main__":
    evaluate_results("backtest_results.csv")
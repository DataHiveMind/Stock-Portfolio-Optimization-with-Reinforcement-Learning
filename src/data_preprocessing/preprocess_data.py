# src/data_preprocessing.py
import pandas as pd
import sqlite3

def preprocess_data(db_path):
    """Preprocesses stock data for RL."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    conn.close()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    returns = df.pct_change().dropna()
    # Add simple technical indicator
    df['SMA_10'] = df.mean(axis=1).rolling(window=10).mean()
    df.dropna(inplace=True)

    return df, returns

if __name__ == "__main__":
    df, returns = preprocess_data('data/database/stock_data.db')
    print("Preprocessed DataFrame:")
    print(df.head())
# src/backtest.py
import pandas as pd
import numpy as np
import tensorflow as tf
import sqlite3

def backtest(model_path, data):
    model = tf.keras.models.load_model(model_path)
    env = TradingEnvironment(data)
    state = env.reset()
    balances = [env.initial_balance]
    # src/backtest.py (continued)
    done = False
    while not done:
        action = np.argmax(model.predict(np.array([state])))
        state, _, done, _ = env.step(action)
        balances.append(env.balance)
    return balances

if __name__ == "__main__":
    conn = sqlite3.connect('data/database/stock_data.db')
    df = pd.read_sql_query("SELECT * FROM stock_data", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)

    balances = backtest("rl_model.h5", df)
    results_df = pd.DataFrame({'Balance': balances})
    results_df.to_csv("backtest_results.csv")
    print("Backtest results saved to backtest_results.csv")
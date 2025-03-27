# src/data_acquisition.py
import pandas as pd
import sqlite3

def load_data_to_sqlite(csv_path, db_path):
    """Loads CSV data into an SQLite database."""
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql('stock_data', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    load_data_to_sqlite('data/raw/stock_data.csv', 'data/database/stock_data.db')
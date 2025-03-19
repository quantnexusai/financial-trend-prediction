import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_stock_data(ticker='AAPL', period='5y', interval='1d', save=True):
    """
    Fetch historical stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol.
    period : str
        Time period to retrieve data for (default: '5y' for 5 years).
    interval : str
        Data interval (default: '1d' for daily).
    save : bool
        Whether to save the data to a CSV file.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing historical stock data.
    """
    print(f"Fetching data for {ticker} over {period} period with {interval} interval...")
    
    # Fetch data
    stock_data = yf.download(ticker, period=period, interval=interval)
    
    # Reset index to have Date as a column
    stock_data = stock_data.reset_index()
    
    # Create a date string for the file name
    date_str = datetime.now().strftime("%Y%m%d")
    
    if save:
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Save data to CSV
        file_path = f"data/{ticker}_{date_str}.csv"
        stock_data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    
    return stock_data

def load_stock_data(file_path):
    """
    Load stock data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing historical stock data.
    """
    return pd.read_csv(file_path, parse_dates=['Date'])

if __name__ == "__main__":
    # Example usage
    aapl_data = fetch_stock_data(ticker='AAPL', period='5y', interval='1d')
    print(f"Data shape: {aapl_data.shape}")
    print(f"Data columns: {aapl_data.columns.tolist()}")
    print(f"Data sample:\n{aapl_data.head()}")
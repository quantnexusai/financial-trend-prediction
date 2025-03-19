import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock price data with OHLCV columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional technical indicator columns.
    """
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # Bollinger Bands (20-day, 2 standard deviations)
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + 2 * data['BB_std']
    data['BB_lower'] = data['BB_middle'] - 2 * data['BB_std']
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Momentum
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
    
    # Price Rate of Change
    data['ROC_5'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5) * 100
    data['ROC_10'] = (data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10) * 100
    
    # Volume-based indicators
    data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Change'] = data['Volume'] / data['Volume'].shift(1) - 1
    
    # Price to Volume Ratio
    data['Price_Volume_Ratio'] = data['Close'] / (data['Volume'] + 1)  # Adding 1 to avoid division by zero
    
    # Drop rows with NaN values (due to rolling windows)
    # data = data.dropna()
    
    return data

def create_target_variable(df, horizon=1):
    """
    Create binary target variable: 1 if price goes up after specified horizon, 0 otherwise.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock price data with 'Close' column.
    horizon : int
        Number of days to look ahead for price movement.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional 'Target' column.
    """
    data = df.copy()
    # Get the current Close price
    current_close = data['Close'].values
    # Get the future Close price (shifted)
    future_close = data['Close'].shift(-horizon).values
    
    # Create Future_Close column
    data['Future_Close'] = data['Close'].shift(-horizon)
    
    # Create Target column using numpy comparison instead of pandas
    import numpy as np
    # Only compare where future_close is not NaN
    target = np.zeros(len(data))
    for i in range(len(data) - horizon):
        target[i] = 1 if future_close[i] > current_close[i] else 0
    
    data['Target'] = target
    
    return data

def create_features_and_target(df, dropna=True, scale_features=True):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock price data with OHLCV columns.
    dropna : bool
        Whether to drop rows with NaN values.
    scale_features : bool
        Whether to scale features using StandardScaler.
        
    Returns:
    --------
    tuple
        (X, y, feature_names) where X is the feature matrix, y is the target vector,
        and feature_names is a list of feature names.
    """
    # Calculate technical indicators
    data = calculate_technical_indicators(df)
    
    # Create target variable
    data = create_target_variable(data)
    
    # Drop date column if it exists
    if 'Date' in data.columns:
        data = data.drop(['Date'], axis=1)
    
    # Drop rows with NaN values if requested
    if dropna:
        data = data.dropna()
    
    # Features and target
    y = data['Target'].copy()
    
    # Drop columns that shouldn't be used as features
    features_to_drop = ['Target', 'Future_Close']
    if 'Adj Close' in data.columns:
        features_to_drop.append('Adj Close')
    X = data.drop(features_to_drop, axis=1)
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    
    return X, y, feature_names

if __name__ == "__main__":
    # Example usage
    from data_acquisition import fetch_stock_data
    
    # Fetch data
    aapl_data = fetch_stock_data(ticker='AAPL', period='5y', interval='1d')
    
    # Process data
    X, y, feature_names = create_features_and_target(aapl_data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of upward movements: {y.sum()}")
    print(f"Number of downward movements: {len(y) - y.sum()}")
    print(f"Feature names: {feature_names}")
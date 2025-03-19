# Financial Trend Prediction

A machine learning project for predicting financial market trends using technical indicators.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange)
![pandas](https://img.shields.io/badge/pandas-2.0.0-green)

## Overview

This project demonstrates the use of machine learning techniques to predict stock price movements based on technical indicators. It showcases an end-to-end machine learning workflow from data acquisition and feature engineering to model training, evaluation, and visualization.

Key components:
- Data acquisition from Yahoo Finance API
- Technical indicator calculation
- Feature engineering
- Gradient Boosting classifier implementation
- Hyperparameter optimization
- Model evaluation and visualization
- Demonstration of a simple trading strategy

## Project Structure

```
financial-trend-prediction/
│
├── data/                        # Stock data storage
├── models/                      # Saved model files
├── notebooks/                   # Jupyter notebooks
│   └── financial_trend_analysis.ipynb
├── src/                         # Source code
│   ├── data_acquisition.py      # Data fetching functions
│   ├── feature_engineering.py   # Feature calculation
│   ├── model_training.py        # Model training and evaluation
│   └── main.py                  # Main script to run pipeline
├── visualizations/              # Generated plots
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/quantnexusai/financial-trend-prediction.git
cd financial-trend-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the main script with default parameters (AAPL stock for 5 years):
```bash
python src/main.py
```

Run with hyperparameter optimization:
```bash
python src/main.py --optimize
```

Customize the run with parameters:
```bash
python src/main.py --ticker MSFT --period 2y --interval 1d --optimize --save-model
```

Available options:
```
  --ticker TICKER       Stock ticker symbol (default: AAPL)
  --period PERIOD       Time period to retrieve data for (default: 5y)
  --interval INTERVAL   Data interval (default: 1d)
  --data-path DATA_PATH Path to CSV file with stock data (optional)
  --keep-na             Keep rows with NaN values (default: False)
  --no-scaling          Disable feature scaling (default: False)
  --test-size TEST_SIZE Proportion of data to use for testing (default: 0.2)
  --optimize            Perform hyperparameter optimization (default: False)
  --cv-folds CV_FOLDS   Number of cross-validation folds (default: 5)
  --save-model          Save model to disk (default: False)
  --model-path MODEL_PATH Path to save model (optional)
```

### Jupyter Notebook

Explore the analysis interactively:
```bash
jupyter notebook notebooks/financial_trend_analysis.ipynb
```

## Features

### Technical Indicators
The project calculates and uses the following technical indicators as features:
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Bollinger Bands
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Momentum indicators
- Volume-based indicators

### Model
- Default: Gradient Boosting Classifier
- Hyperparameter optimization via Grid Search
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

### Visualizations
- Feature importance
- ROC curve
- Confusion matrix
- Price and technical indicator plots
- Cumulative returns comparison

## Results

### Default Model Performance
When run on Apple (AAPL) stock data with default parameters:
- Accuracy: 47.93%
- Precision: 51.37%
- Recall: 57.69%
- F1 Score: 54.35%

### Optimized Model Performance
After hyperparameter optimization:
- Best Parameters: 
  - learning_rate: 0.01
  - max_depth: 2
  - min_samples_split: 2
  - n_estimators: 50
  - subsample: 0.8
- Accuracy: 52.89%
- Precision: 53.42%
- Recall: 96.15%
- F1 Score: 68.68%

The optimized model shows a significant improvement in recall (capturing 96.15% of actual price increases) and F1 score, though with a trade-off in precision. The model demonstrates a strong bias toward predicting price increases, which could be beneficial in certain trading strategies but may lead to more false positives.

*Note: Actual results may vary depending on the time period and specific market conditions.*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Ari Harrison - ari@quantnexus.ai

Project Link: https://github.com/quantnexusai/financial-trend-prediction

## Disclaimer

This project is for educational purposes only. The predictions made by this model should not be used as financial advice. Trading and investing in financial markets involves risk, and past performance does not guarantee future results.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing the financial data API
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [pandas](https://pandas.pydata.org/) for data manipulation
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualizations
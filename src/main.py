import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from data_acquisition import fetch_stock_data, load_stock_data
from feature_engineering import create_features_and_target
from model_training import (
    train_model, evaluate_model, optimize_hyperparameters,
    plot_feature_importance, plot_roc_curve, save_model, create_pipeline
)

def main(args):
    """
    Main function to run the financial trend prediction pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments.
    """
    print(f"Starting financial trend prediction for {args.ticker}...")
    
    # Step 1: Data acquisition
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}...")
        data = load_stock_data(args.data_path)
    else:
        print(f"Fetching data for {args.ticker}...")
        data = fetch_stock_data(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            save=True
        )
    
    print(f"Data shape: {data.shape}")
    
    # Step 2: Feature engineering
    print("\nCreating features and target variable...")
    X, y, feature_names = create_features_and_target(
        data,
        dropna=not args.keep_na,
        scale_features=not args.no_scaling
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Model training
    print("\nTraining model...")
    if args.optimize:
        print("Performing hyperparameter optimization...")
        # Split data for optimization
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        
        # Optimize hyperparameters
        grid_search = optimize_hyperparameters(X_train, y_train, cv=args.cv_folds)
        best_params = grid_search.best_params_
        
        # Create and train pipeline with best parameters
        pipeline = create_pipeline(best_params)
        pipeline.fit(X_train, y_train)
        model = pipeline
    else:
        # Train with default parameters
        model, X_train, X_test, y_train, y_test = train_model(
            X, y, test_size=args.test_size
        )
    
    # Step 4: Model evaluation
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 5: Visualizations
    print("\nGenerating visualizations...")
    
    # Feature importance plot
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        plot_feature_importance(model.named_steps['classifier'], feature_names)
    else:
        print("Model doesn't have feature_importances_ attribute. Skipping feature importance plot.")
    
    # ROC curve plot
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['roc_auc']
    )
    
    # Step 6: Save model
    if args.save_model:
        save_model(model, args.model_path)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Financial Trend Prediction')
    
    # Data acquisition arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--period', type=str, default='5y',
                        help='Time period to retrieve data for (default: 5y)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (default: 1d)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV file with stock data (optional)')
    
    # Feature engineering arguments
    parser.add_argument('--keep-na', action='store_true',
                        help='Keep rows with NaN values (default: False)')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable feature scaling (default: False)')
    
    # Model training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization (default: False)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    
    # Model saving arguments
    parser.add_argument('--save-model', action='store_true',
                        help='Save model to disk (default: False)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to save model (optional)')
    
    args = parser.parse_args()
    
    # Run main function
    main(args)
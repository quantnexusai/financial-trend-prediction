import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Gradient Boosting Classifier on the provided data.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion of the data to use for testing.
    random_state : int
        Random seed for reproducibility.
        
    Returns:
    --------
    tuple
        (model, X_train, X_test, y_train, y_test) containing the trained model
        and the train/test split data.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    
    # Create and train the model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target variable.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Print results
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = roc_auc
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()
    
    return metrics

def optimize_hyperparameters(X_train, y_train, cv=5):
    """
    Perform grid search to find optimal hyperparameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target variable.
    cv : int
        Number of cross-validation folds.
        
    Returns:
    --------
    GridSearchCV
        Fitted grid search object.
    """
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Create base model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Setup grid search
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )
    
    # Fit grid search
    print("Performing grid search for hyperparameter optimization...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    model : estimator
        Trained model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to display.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top_n features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_importances)), top_importances, align='center')
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Features by Importance')
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/feature_importance.png')
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : array
        False positive rates.
    tpr : array
        True positive rates.
    roc_auc : float
        Area under the ROC curve.
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/roc_curve.png')
    
    return fig

def save_model(model, filename=None):
    """
    Save model to disk.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    filename : str, optional
        Filename to save model to. If None, a default name with timestamp is used.
        
    Returns:
    --------
    str
        Path to saved model.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/gb_model_{timestamp}.pkl"
    
    # Save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filename}")
    
    return filename

def create_pipeline(best_params=None):
    """
    Create a scikit-learn pipeline with preprocessing and model.
    
    Parameters:
    -----------
    best_params : dict, optional
        Dictionary of best hyperparameters for the model.
        
    Returns:
    --------
    Pipeline
        Scikit-learn pipeline.
    """
    # Default parameters if none provided
    if best_params is None:
        best_params = {
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'n_estimators': 100,
            'subsample': 0.9
        }
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            n_estimators=best_params['n_estimators'],
            subsample=best_params['subsample'],
            random_state=42
        ))
    ])
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    from data_acquisition import fetch_stock_data
    from feature_engineering import create_features_and_target
    
    # Fetch data
    aapl_data = fetch_stock_data(ticker='AAPL', period='5y', interval='1d')
    
    # Process data
    X, y, feature_names = create_features_and_target(aapl_data, scale_features=False)
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Plot ROC curve
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['roc_auc']
    )
    
    # Save model
    save_model(model)
    
    # Optional: Hyperparameter optimization
    # grid_search = optimize_hyperparameters(X_train, y_train)
    # best_model = grid_search.best_estimator_
    # save_model(best_model, 'models/gb_model_optimized.pkl')
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


def load_and_preprocess_data():
    """Load molecular embeddings and preprocess them."""
    print("Loading molecular embeddings...")
    actives = pickle.load(open('/home/jderiz/EDDDe/data/qsar/embeddings/AID435034/actives/preds_extended.pkl', 'rb'))
    inactives = pickle.load(open('/home/jderiz/EDDDe/data/qsar/embeddings/AID435034/inactives/preds_extended.pkl', 'rb'))
    
    print(f"Loaded {len(actives)} active compounds and {len(inactives)} inactive compounds")
    
    # Compute mean embeddings along atom axis
    actives_mean = [np.mean(embed, axis=0) for embed in actives.values()]
    inactives_mean = [np.mean(embed, axis=0) for embed in inactives.values()]
    
    # Combine data
    X = np.concatenate([actives_mean, inactives_mean])
    y = np.concatenate([np.ones(len(actives_mean)), np.zeros(len(inactives_mean))])
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X, y


def comprehensive_cross_validation(X, y, model, cv_folds=5):
    """Perform comprehensive cross-validation with all classification metrics."""
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    # Initialize cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Initialize metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'average_precision': []
    }
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        metrics['average_precision'].append(average_precision_score(y_test, y_proba))
        
        print(f"Fold {fold + 1}: Accuracy={metrics['accuracy'][-1]:.4f}, "
              f"F1={metrics['f1'][-1]:.4f}, ROC-AUC={metrics['roc_auc'][-1]:.4f}")
    
    # Calculate mean and std for each metric
    results = {}
    for metric_name, values in metrics.items():
        results[f'{metric_name}_mean'] = np.mean(values)
        results[f'{metric_name}_std'] = np.std(values)
    
    return results, metrics


def grid_search_optimization(X, y, cv_folds=5):
    """Perform grid search to find optimal hyperparameters."""
    print("\nPerforming grid search for hyperparameter optimization...")
    
    # Define parameter grid
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # gamma is used for rbf and poly kernels
    }
    
    # Initialize SVM with probability=True for ROC-AUC calculation
    svm = SVC(probability=True, random_state=42)
    
    # Initialize cross-validation splitter
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        svm, param_grid, cv=cv_splitter, 
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Scale features before grid search
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit grid search
    grid_search.fit(X_scaled, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, scaler


def evaluate_best_model(X, y, best_model, scaler, cv_folds=5):
    """Evaluate the best model with comprehensive metrics."""
    print("\nEvaluating best model with comprehensive metrics...")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Perform comprehensive cross-validation
    results, fold_metrics = comprehensive_cross_validation(X_scaled, y, best_model, cv_folds)
    
    # Print summary results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
        mean_val = results[f'{metric}_mean']
        std_val = results[f'{metric}_std']
        print(f"{metric.upper():20}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results, fold_metrics


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve for the final model."""
    print("\nGenerating ROC curve...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()
    
    return fpr, tpr, roc_auc


def save_model_and_scaler(model, scaler, best_params, results, save_dir):
    """Save the trained model, scaler, parameters, and results."""
    print(f"\nSaving model and results to: {save_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the trained model
    model_path = os.path.join(save_dir, f'svm_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save parameters and results
    results_path = os.path.join(save_dir, f'model_results_{timestamp}.pkl')
    save_data = {
        'best_parameters': best_params,
        'cross_validation_results': results,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'timestamp': timestamp
    }
    with open(results_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Results saved to: {results_path}")
    
    return model_path, scaler_path, results_path


def main():
    """Main function to run the complete SVM analysis."""
    print("Starting SVM Analysis with Cross-Validation and Grid Search")
    print("="*60)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Perform grid search optimization
    best_model, best_params, scaler = grid_search_optimization(X, y)
    
    # Evaluate best model
    results, fold_metrics = evaluate_best_model(X, y, best_model, scaler)
    
    # Final evaluation on full dataset for detailed metrics
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    X_scaled = scaler.transform(X)
    y_pred = best_model.predict(X_scaled)
    y_proba = best_model.predict_proba(X_scaled)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print(f"\nFinal ROC-AUC Score: {roc_auc_score(y, y_proba):.4f}")
    print(f"Final Average Precision Score: {average_precision_score(y, y_proba):.4f}")
    
    # Plot ROC curve
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    roc_save_path = f'/home/jderiz/EDDDe/data/qsar/embeddings/AID435034/roc_curve_{timestamp}.png'
    fpr, tpr, roc_auc = plot_roc_curve(y, y_proba, save_path=roc_save_path)
    
    # Save model and results to parent folder
    save_dir = '/home/jderiz/EDDDe/data/qsar/embeddings/AID435034'
    model_path, scaler_path, results_path = save_model_and_scaler(
        best_model, scaler, best_params, results, save_dir
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"ROC curve saved to: {roc_save_path}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Results saved to: {results_path}")
    
    return best_model, best_params, results


if __name__ == "__main__":
    best_model, best_params, results = main()
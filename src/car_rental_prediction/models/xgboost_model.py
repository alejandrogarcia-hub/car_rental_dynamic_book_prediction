"""
XGBoost Model for Book or Wait Decision
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


def create_price_features(rental_prices_df, competitor_prices_df):
    """Create features for the book/wait decision model."""
    features_list = []
    
    # Get unique combinations of supplier, location, car_class for price lookups
    price_keys = rental_prices_df[['supplier_id', 'location_id', 'car_class']].drop_duplicates()
    
    for _, key in price_keys.iterrows():
        supplier_id = key['supplier_id']
        location_id = key['location_id']
        car_class = key['car_class']
        
        # Get price history for this combination
        mask = (rental_prices_df['supplier_id'] == supplier_id) & \
               (rental_prices_df['location_id'] == location_id) & \
               (rental_prices_df['car_class'] == car_class)
        
        price_history = rental_prices_df[mask].sort_values('date')
        
        # For each date, calculate features and target
        for i in range(7, len(price_history) - 7):  # Need 7 days before and after
            current_date = price_history.iloc[i]['date']
            current_price = price_history.iloc[i]['current_price']
            
            # Calculate price change features
            price_1d_ago = price_history.iloc[i-1]['current_price']
            price_3d_ago = price_history.iloc[i-3]['current_price']
            price_7d_ago = price_history.iloc[i-7]['current_price']
            
            # Future price (for target calculation)
            future_prices = price_history.iloc[i+1:i+8]['current_price'].values
            max_future_price = np.max(future_prices)
            
            # Calculate target: 1 if should book now, 0 if should wait
            # Book now if current price is lower than max price in next 7 days
            should_book = 1 if current_price < max_future_price else 0
            
            # Calculate features
            features = {
                'date': current_date,
                'supplier_id': supplier_id,
                'location_id': location_id,
                'car_class': car_class,
                'current_price': current_price,
                'price_change_1d': (current_price - price_1d_ago) / price_1d_ago,
                'price_change_3d': (current_price - price_3d_ago) / price_3d_ago,
                'price_change_7d': (current_price - price_7d_ago) / price_7d_ago,
                'price_volatility_7d': price_history.iloc[i-7:i]['current_price'].std() / price_history.iloc[i-7:i]['current_price'].mean(),
                'day_of_week': current_date.dayofweek,
                'month': current_date.month,
                'is_weekend': 1 if current_date.dayofweek >= 5 else 0,
                'days_until_rental': 14,  # Default booking window
                'target': should_book
            }
            
            # Add more advanced features for XGBoost
            # Price percentile within historical range
            hist_prices = price_history.iloc[max(0, i-30):i]['current_price'].values
            if len(hist_prices) > 0:
                features['price_percentile'] = (current_price - np.min(hist_prices)) / (np.max(hist_prices) - np.min(hist_prices) + 1e-6)
            else:
                features['price_percentile'] = 0.5
            
            # Price trend (linear regression slope)
            if len(hist_prices) >= 3:
                x = np.arange(len(hist_prices))
                slope = np.polyfit(x, hist_prices, 1)[0]
                features['price_trend'] = slope
            else:
                features['price_trend'] = 0
            
            # Seasonal features
            features['is_peak_season'] = 1 if current_date.month in [6, 7, 8, 12] else 0
            features['quarter'] = (current_date.month - 1) // 3 + 1
            
            # Add competitor price features
            comp_mask = (competitor_prices_df['location_id'] == location_id) & \
                       (competitor_prices_df['car_class'] == car_class) & \
                       (competitor_prices_df['date'] == current_date)
            
            comp_prices = competitor_prices_df[comp_mask]['comp_min_price'].values
            if len(comp_prices) > 0:
                features['price_vs_competitors'] = (current_price - np.mean(comp_prices)) / np.mean(comp_prices)
                features['is_cheapest'] = 1 if current_price < np.min(comp_prices) else 0
                features['price_rank'] = np.sum(current_price >= comp_prices) / len(comp_prices)
            else:
                features['price_vs_competitors'] = 0
                features['is_cheapest'] = 0
                features['price_rank'] = 0.5
            
            features_list.append(features)
    
    return pd.DataFrame(features_list)


def evaluate_model(y_true, y_pred, y_prob, model_name="XGBoost"):
    """Evaluate model performance with comprehensive metrics."""
    
    print(f"\n{model_name} Model Evaluation:")
    print("=" * 50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Wait', 'Book Now']))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Wait', 'Book Now'], 
                yticklabels=['Wait', 'Book Now'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc


def plot_feature_importance(model, feature_columns, top_n=15):
    """Plot feature importance from XGBoost model."""
    
    # Get feature importance
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in importance_dict.items()
    ]).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Weight)')
    plt.title(f'XGBoost Feature Importance - Top {top_n} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {top_n} Most Important Features:")
    print(top_features[['feature', 'importance']].to_string(index=False))
    
    return importance_df


def main():
    print("XGBoost Model for Book or Wait Decision")
    print("=" * 50)
    
    # Load data
    data_dir = Path('data/synthetic_data')
    
    print("Loading synthetic data...")
    users_df = pd.read_csv(data_dir / 'synthetic_users.csv')
    searches_df = pd.read_csv(data_dir / 'synthetic_searches.csv')
    bookings_df = pd.read_csv(data_dir / 'synthetic_bookings.csv')
    rental_prices_df = pd.read_csv(data_dir / 'synthetic_rental_prices.csv')
    competitor_prices_df = pd.read_csv(data_dir / 'synthetic_competitor_prices.csv')
    
    # Convert timestamps
    searches_df['search_ts'] = pd.to_datetime(searches_df['search_ts'])
    bookings_df['booking_ts'] = pd.to_datetime(bookings_df['booking_ts'])
    rental_prices_df['date'] = pd.to_datetime(rental_prices_df['obs_ts'])
    competitor_prices_df['date'] = pd.to_datetime(competitor_prices_df['obs_date'])
    
    print(f"Users: {users_df.shape}")
    print(f"Searches: {searches_df.shape}")
    print(f"Bookings: {bookings_df.shape}")
    print(f"Rental Prices: {rental_prices_df.shape}")
    print(f"Competitor Prices: {competitor_prices_df.shape}")
    
    # Create features
    print("\nCreating features...")
    features_df = create_price_features(rental_prices_df, competitor_prices_df)
    print(f"Created {len(features_df)} training examples")
    print(f"Target distribution:\n{features_df['target'].value_counts(normalize=True)}")
    
    # Prepare features
    feature_columns = ['price_change_1d', 'price_change_3d', 'price_change_7d', 
                       'price_volatility_7d', 'day_of_week', 'month', 'is_weekend',
                       'days_until_rental', 'price_vs_competitors', 'is_cheapest',
                       'supplier_id', 'location_id', 'price_percentile', 'price_trend',
                       'is_peak_season', 'quarter', 'price_rank']
    
    # Map car_class to numeric values
    car_class_map = {'economy': 0, 'compact': 1, 'suv': 2}
    features_df['car_class_num'] = features_df['car_class'].map(car_class_map)
    feature_columns.append('car_class_num')
    
    X = features_df[feature_columns]
    y = features_df['target']
    
    # Time-based train/test split
    split_date = features_df['date'].quantile(0.8)
    train_mask = features_df['date'] < split_date
    test_mask = ~train_mask
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Check for NaN/inf values
    print(f"\nChecking for invalid values in features...")
    print(f"NaN in X_train: {X_train.isna().sum().sum()}")
    print(f"Inf in X_train: {np.isinf(X_train.values).sum()}")
    
    # Replace NaN and inf with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Scale weights to handle class imbalance
    class_weights = {0: len(y_train) / (2 * (y_train == 0).sum()),
                     1: len(y_train) / (2 * (y_train == 1).sum())}
    sample_weights = y_train.map(class_weights)
    
    print(f"\nClass weights: {class_weights}")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    # Train model
    model.fit(X_train, y_train, 
              sample_weight=sample_weights)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    roc_auc = evaluate_model(y_test, y_pred, y_prob, "XGBoost")
    
    # Plot feature importance
    importance_df = plot_feature_importance(model, feature_columns)
    
    # Calculate business metrics
    print("\nBusiness Impact Analysis:")
    print("=" * 30)
    
    test_data = features_df[test_mask].copy()
    test_data['prediction'] = y_pred
    test_data['probability'] = y_prob
    
    # Calculate potential savings/losses
    total_impact = 0
    correct_book_decisions = 0
    total_book_opportunities = 0
    
    for _, row in test_data.iterrows():
        current_price = row['current_price']
        actual_decision = row['target']
        predicted_decision = row['prediction']
        
        if actual_decision == 1:  # Should have booked
            total_book_opportunities += 1
            if predicted_decision == 1:  # Correctly predicted book
                correct_book_decisions += 1
                total_impact += current_price * 0.1  # Assume 10% price increase
            else:  # Missed opportunity
                total_impact -= current_price * 0.1
    
    accuracy_on_book_opportunities = correct_book_decisions / total_book_opportunities if total_book_opportunities > 0 else 0
    
    print(f"Total potential impact: ${total_impact:,.2f}")
    print(f"Book opportunity accuracy: {accuracy_on_book_opportunities:.2%}")
    print(f"Average impact per decision: ${total_impact / len(test_data):,.2f}")
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save XGBoost model
    model.save_model(str(model_dir / 'xgboost_model.json'))
    
    # Save feature columns and other metadata
    joblib.dump({
        'feature_columns': feature_columns,
        'car_class_map': car_class_map,
        'class_weights': class_weights,
        'roc_auc': roc_auc,
        'feature_importance': importance_df
    }, model_dir / 'xgboost_metadata.pkl')
    
    print(f"\nModel saved to {model_dir}")
    print(f"Final ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
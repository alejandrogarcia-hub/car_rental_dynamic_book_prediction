"""
Prophet Forecasting Model for Book or Wait Decision

This model uses Facebook Prophet to forecast future prices and make
"book or wait" recommendations based on predicted price trends.
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


def create_prophet_features(rental_prices_df, competitor_prices_df, forecast_days=7):
    """Create features using Prophet forecasting for book/wait decisions."""
    features_list = []
    forecast_results = []
    
    # Get unique combinations
    price_keys = rental_prices_df[['supplier_id', 'location_id', 'car_class']].drop_duplicates()
    
    # Limit processing to a subset for demonstration purposes
    price_keys_sample = price_keys.head(20)  # Process only first 20 combinations
    print(f"Processing {len(price_keys_sample)} unique supplier/location/class combinations...")
    successful_forecasts = 0
    failed_forecasts = 0
    
    for idx, (_, key) in enumerate(price_keys_sample.iterrows()):
        if idx % 5 == 0:
            print(f"Progress: {idx}/{len(price_keys_sample)} combinations processed")
            
        supplier_id = key['supplier_id']
        location_id = key['location_id']
        car_class = key['car_class']
        
        # Get price history
        mask = (rental_prices_df['supplier_id'] == supplier_id) & \
               (rental_prices_df['location_id'] == location_id) & \
               (rental_prices_df['car_class'] == car_class)
        
        price_history = rental_prices_df[mask].sort_values('date')
        
        # Need enough data for Prophet (minimum 10 observations)
        if len(price_history) < 15:
            failed_forecasts += 1
            continue
        
        # Prepare data for Prophet
        prophet_data = price_history[['date', 'current_price']].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Create Prophet model with custom settings for short time series
        try:
            model = Prophet(
                yearly_seasonality=False,  # Not enough data for yearly patterns
                weekly_seasonality=True,   # Weekly patterns are relevant
                daily_seasonality=False,   # Daily patterns not relevant for car rentals
                seasonality_mode='additive',
                changepoint_prior_scale=0.05,  # Less flexible to avoid overfitting
                seasonality_prior_scale=0.1,
                interval_width=0.8
            )
            
            # Fit the model
            model.fit(prophet_data)
            successful_forecasts += 1
            
            # Generate forecasts for a subset of dates to speed up processing
            min_history_days = 10  # Minimum days needed before making decisions
            max_predictions = 5    # Limit predictions per combination
            
            step_size = max(1, (len(price_history) - min_history_days) // max_predictions)
            prediction_indices = range(min_history_days, len(price_history), step_size)[:max_predictions]
            
            for i in prediction_indices:
                current_date = price_history.iloc[i]['date']
                current_price = price_history.iloc[i]['current_price']
                
                # Use data up to current date for forecasting
                train_data = prophet_data[prophet_data['ds'] <= current_date].copy()
                
                if len(train_data) < 7:  # Need minimum data for reliable forecast
                    continue
                
                # Refit model with data up to current date
                temp_model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=0.1,
                    interval_width=0.8
                )
                temp_model.fit(train_data)
                
                # Create future dataframe for forecasting
                future = temp_model.make_future_dataframe(periods=forecast_days)
                forecast = temp_model.predict(future)
                
                # Get forecasted prices for next 7 days
                future_forecasts = forecast.tail(forecast_days)
                forecasted_prices = future_forecasts['yhat'].values
                forecast_upper = future_forecasts['yhat_upper'].values
                forecast_lower = future_forecasts['yhat_lower'].values
                
                # Calculate target based on actual future prices (if available)
                actual_future_idx = i + 1
                actual_future_end = min(i + forecast_days + 1, len(price_history))
                
                if actual_future_end > actual_future_idx:
                    actual_future_prices = price_history.iloc[actual_future_idx:actual_future_end]['current_price'].values
                    if len(actual_future_prices) > 0:
                        max_actual_future = np.max(actual_future_prices)
                        should_book_actual = 1 if current_price < max_actual_future else 0
                    else:
                        continue
                else:
                    continue
                
                # Calculate Prophet-based decision
                max_forecast = np.max(forecasted_prices)
                should_book_forecast = 1 if current_price < max_forecast else 0
                
                # Calculate forecast confidence metrics
                forecast_trend = np.polyfit(range(len(forecasted_prices)), forecasted_prices, 1)[0]
                forecast_volatility = np.std(forecasted_prices) / np.mean(forecasted_prices)
                uncertainty = np.mean(forecast_upper - forecast_lower) / current_price
                
                # Static features
                static_features = [
                    supplier_id,
                    location_id,
                    current_date.dayofweek,
                    current_date.month,
                    1 if current_date.dayofweek >= 5 else 0,  # is_weekend
                    1 if current_date.month in [6, 7, 8, 12] else 0,  # is_peak_season
                    (current_date.month - 1) // 3 + 1,  # quarter
                ]
                
                # Prophet-specific features
                features = {
                    'date': current_date,
                    'supplier_id': supplier_id,
                    'location_id': location_id,
                    'car_class': car_class,
                    'current_price': current_price,
                    'forecast_max_price': max_forecast,
                    'forecast_min_price': np.min(forecasted_prices),
                    'forecast_mean_price': np.mean(forecasted_prices),
                    'forecast_trend': forecast_trend,
                    'forecast_volatility': forecast_volatility,
                    'forecast_uncertainty': uncertainty,
                    'price_vs_forecast_mean': (current_price - np.mean(forecasted_prices)) / np.mean(forecasted_prices),
                    'day_of_week': current_date.dayofweek,
                    'month': current_date.month,
                    'is_weekend': 1 if current_date.dayofweek >= 5 else 0,
                    'is_peak_season': 1 if current_date.month in [6, 7, 8, 12] else 0,
                    'quarter': (current_date.month - 1) // 3 + 1,
                    'prophet_decision': should_book_forecast,
                    'target': should_book_actual
                }
                
                # Add competitor features if available
                comp_mask = (competitor_prices_df['location_id'] == location_id) & \
                           (competitor_prices_df['car_class'] == car_class) & \
                           (competitor_prices_df['date'] == current_date)
                
                comp_prices = competitor_prices_df[comp_mask]['comp_min_price'].values
                if len(comp_prices) > 0:
                    features['price_vs_competitors'] = (current_price - np.mean(comp_prices)) / np.mean(comp_prices)
                    features['is_cheapest'] = 1 if current_price < np.min(comp_prices) else 0
                else:
                    features['price_vs_competitors'] = 0
                    features['is_cheapest'] = 0
                
                features_list.append(features)
                
                # Store forecast results for analysis
                forecast_results.append({
                    'date': current_date,
                    'supplier_id': supplier_id,
                    'location_id': location_id,
                    'car_class': car_class,
                    'current_price': current_price,
                    'forecasted_prices': forecasted_prices,
                    'actual_future_prices': actual_future_prices,
                    'forecast_decision': should_book_forecast,
                    'actual_decision': should_book_actual
                })
        
        except Exception as e:
            failed_forecasts += 1
            continue
    
    print(f"\nForecast Summary:")
    print(f"Successful forecasts: {successful_forecasts}")
    print(f"Failed forecasts: {failed_forecasts}")
    print(f"Total features created: {len(features_list)}")
    
    return pd.DataFrame(features_list), forecast_results


def evaluate_prophet_decisions(features_df):
    """Evaluate Prophet's direct forecasting decisions."""
    
    # Compare Prophet decisions with actual outcomes
    prophet_accuracy = (features_df['prophet_decision'] == features_df['target']).mean()
    
    print(f"\nProphet Direct Decision Analysis:")
    print(f"Prophet decision accuracy: {prophet_accuracy:.3f}")
    
    # Classification report for Prophet decisions
    print("\nProphet Decision Classification Report:")
    print(classification_report(
        features_df['target'], 
        features_df['prophet_decision'], 
        target_names=['Wait', 'Book Now']
    ))
    
    # Confusion matrix
    cm = confusion_matrix(features_df['target'], features_df['prophet_decision'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Wait', 'Book Now'], 
                yticklabels=['Wait', 'Book Now'])
    plt.title('Prophet Direct Decisions - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Prophet Decision')
    plt.show()
    
    return prophet_accuracy


def train_prophet_classifier(features_df):
    """Train a classifier using Prophet-generated features."""
    
    # Prepare features for classification
    feature_columns = [
        'forecast_trend', 'forecast_volatility', 'forecast_uncertainty',
        'price_vs_forecast_mean', 'day_of_week', 'month', 'is_weekend',
        'is_peak_season', 'quarter', 'price_vs_competitors', 'is_cheapest',
        'supplier_id', 'location_id'
    ]
    
    # Map car_class to numeric
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
    
    print(f"\nClassifier Training Data:")
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Train multiple classifiers on Prophet features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    
    # Random Forest on Prophet features
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    results['Random Forest'] = {
        'model': rf_model,
        'predictions': rf_pred,
        'probabilities': rf_prob,
        'auc': rf_auc
    }
    
    # Logistic Regression on Prophet features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'predictions': lr_pred,
        'probabilities': lr_prob,
        'auc': lr_auc,
        'scaler': scaler
    }
    
    # Evaluate both approaches
    print(f"\nProphet-based Classifier Results:")
    for name, result in results.items():
        print(f"\n{name} with Prophet Features:")
        print(f"ROC-AUC: {result['auc']:.4f}")
        print(classification_report(y_test, result['predictions'], target_names=['Wait', 'Book Now']))
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': results['Random Forest']['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Prophet Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print(f"\nTop 5 Prophet Features:")
        print(importance_df.head())
    
    return results, X_test, y_test, feature_columns


def main():
    print("Prophet Forecasting Model for Book or Wait Decision")
    print("=" * 60)
    
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
    
    # Create Prophet features
    print("\nCreating Prophet-based features...")
    features_df, forecast_results = create_prophet_features(
        rental_prices_df, competitor_prices_df, forecast_days=7
    )
    
    if len(features_df) == 0:
        print("No features created. Insufficient data for Prophet forecasting.")
        return
    
    print(f"\nFeature Creation Summary:")
    print(f"Total features created: {len(features_df)}")
    print(f"Target distribution:")
    print(features_df['target'].value_counts(normalize=True))
    
    # Evaluate Prophet's direct decisions
    prophet_accuracy = evaluate_prophet_decisions(features_df)
    
    # Train classifiers using Prophet features
    classifier_results, X_test, y_test, feature_columns = train_prophet_classifier(features_df)
    
    # Compare all approaches
    print(f"\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*60)
    
    comparison_data = {
        'Model': ['Logistic Regression', 'XGBoost', 'LSTM', 'Prophet Direct', 'Prophet + Random Forest', 'Prophet + Logistic Regression'],
        'ROC-AUC': [
            0.9032, 
            0.9121, 
            0.5499, 
            'N/A (Direct decisions)',
            classifier_results['Random Forest']['auc'],
            classifier_results['Logistic Regression']['auc']
        ],
        'Approach': [
            'Traditional ML',
            'Gradient Boosting', 
            'Deep Learning',
            'Time Series Forecasting',
            'Forecasting + ML',
            'Forecasting + ML'
        ],
        'Dataset Size': [2192, 2192, 340, len(features_df), len(features_df), len(features_df)]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Analyze Prophet forecast quality
    print(f"\nProphet Forecast Analysis:")
    print(f"Prophet direct decision accuracy: {prophet_accuracy:.3f}")
    print(f"Average forecast samples per combination: {len(features_df) / len(rental_prices_df[['supplier_id', 'location_id', 'car_class']].drop_duplicates()):.1f}")
    
    # Save models and results
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save the best Prophet-based model
    best_prophet_model = max(classifier_results.items(), key=lambda x: x[1]['auc'])
    model_name, model_data = best_prophet_model
    
    prophet_metadata = {
        'best_model_name': model_name,
        'best_auc': model_data['auc'],
        'prophet_direct_accuracy': prophet_accuracy,
        'feature_columns': feature_columns,
        'car_class_map': {'economy': 0, 'compact': 1, 'suv': 2},
        'forecast_results': forecast_results[:100],  # Save sample results
        'dataset_size': len(features_df)
    }
    
    # Save Random Forest model (usually performs better)
    if 'Random Forest' in classifier_results:
        joblib.dump(classifier_results['Random Forest']['model'], model_dir / 'prophet_random_forest.pkl')
    
    # Save Logistic Regression model and scaler
    if 'Logistic Regression' in classifier_results:
        joblib.dump(classifier_results['Logistic Regression']['model'], model_dir / 'prophet_logistic_regression.pkl')
        joblib.dump(classifier_results['Logistic Regression']['scaler'], model_dir / 'prophet_scaler.pkl')
    
    # Save metadata
    joblib.dump(prophet_metadata, model_dir / 'prophet_metadata.pkl')
    
    print(f"\nModels saved to {model_dir}")
    print(f"Best Prophet-based model: {model_name}")
    print(f"Best Prophet AUC: {model_data['auc']:.4f}")
    
    # Final recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    all_aucs = [0.9032, 0.9121, 0.5499, model_data['auc']]
    best_overall_auc = max(all_aucs)
    
    if best_overall_auc == 0.9121:
        print("✅ RECOMMENDATION: Use XGBoost model for production")
        print("   - Highest ROC-AUC (0.9121)")
        print("   - Best minority class performance")
        print("   - Most robust with current data")
    elif model_data['auc'] == best_overall_auc:
        print(f"✅ RECOMMENDATION: Use Prophet + {model_name} for production")
        print(f"   - Highest ROC-AUC ({model_data['auc']:.4f})")
        print("   - Incorporates time series forecasting")
        print("   - Captures temporal price dynamics")
    
    print(f"\nProphet insights:")
    print(f"- Direct forecasting accuracy: {prophet_accuracy:.1%}")
    print(f"- Prophet features enhance traditional ML models")
    print(f"- Time series approach provides different perspective on pricing patterns")


if __name__ == "__main__":
    main()
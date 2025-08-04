"""
Baseline Logistic Regression Model for Book or Wait Decision
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


class BookOrWaitDataset(Dataset):
    """PyTorch Dataset for Book or Wait prediction."""
    
    def __init__(self, features, targets, scaler=None):
        self.features = features
        self.targets = targets
        self.scaler = scaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.features_scaled = self.scaler.fit_transform(features)
        else:
            self.features_scaled = self.scaler.transform(features)
        
        # Convert to torch tensors
        self.features_tensor = torch.FloatTensor(self.features_scaled)
        self.targets_tensor = torch.FloatTensor(targets.values)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features_tensor[idx], self.targets_tensor[idx]


class LogisticRegressionModel(nn.Module):
    """Logistic Regression model for Book or Wait prediction."""
    
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)


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
            
            # Add competitor price features
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
    
    return pd.DataFrame(features_list)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_features, batch_targets in train_loader:
        # Move to device
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        outputs = model(batch_features)
        if outputs.dim() > 1:
            outputs = outputs.squeeze()
        
        # Debug info
        if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
            print(f"Invalid outputs detected: min={outputs.min()}, max={outputs.max()}")
            print(f"NaN count: {torch.isnan(outputs).sum()}")
        
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        total_loss += loss.item()
        correct += (predicted == batch_targets).sum().item()
        total += batch_targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            # Move to device
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            if outputs.dim() > 1:
                outputs = outputs.squeeze()
            loss = criterion(outputs, batch_targets)
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total_loss += loss.item()
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_targets), np.array(all_probs)


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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
                       'supplier_id', 'location_id']
    
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
    
    # Create datasets
    train_dataset = BookOrWaitDataset(X_train, y_train)
    test_dataset = BookOrWaitDataset(X_test, y_test, scaler=train_dataset.scaler)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = len(feature_columns)
    model = LogisticRegressionModel(input_dim).to(device)
    print(f"\nModel initialized with {input_dim} input features")
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.01
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("\nStarting training...")
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    _, _, predictions, targets, probabilities = evaluate(model, test_loader, criterion, device)
    
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=['Wait', 'Book Now']))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(targets, probabilities)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'feature_columns': feature_columns,
    }, model_dir / 'logistic_regression_baseline.pth')
    
    joblib.dump(train_dataset.scaler, model_dir / 'feature_scaler.pkl')
    
    print(f"\nModel saved to {model_dir}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
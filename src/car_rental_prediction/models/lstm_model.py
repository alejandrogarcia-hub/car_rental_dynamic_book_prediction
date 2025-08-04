"""
LSTM Model for Book or Wait Decision
This model uses sequential price patterns to make predictions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


class PriceSequenceDataset(Dataset):
    """PyTorch Dataset for sequential price data."""
    
    def __init__(self, sequences, targets, static_features=None, scaler=None, sequence_scaler=None):
        self.sequences = sequences
        self.targets = targets
        self.static_features = static_features
        self.scaler = scaler
        self.sequence_scaler = sequence_scaler
        
        # Scale static features
        if static_features is not None:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.static_scaled = self.scaler.fit_transform(static_features)
            else:
                self.static_scaled = self.scaler.transform(static_features)
        else:
            self.static_scaled = None
        
        # Scale sequences
        if self.sequence_scaler is None:
            self.sequence_scaler = MinMaxScaler()
            # Reshape for scaling: (samples * timesteps, features)
            original_shape = sequences.shape
            sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
            sequences_scaled = self.sequence_scaler.fit_transform(sequences_reshaped)
            self.sequences_scaled = sequences_scaled.reshape(original_shape)
        else:
            original_shape = sequences.shape
            sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
            sequences_scaled = self.sequence_scaler.transform(sequences_reshaped)
            self.sequences_scaled = sequences_scaled.reshape(original_shape)
        
        # Convert to tensors
        self.sequences_tensor = torch.FloatTensor(self.sequences_scaled)
        self.targets_tensor = torch.FloatTensor(targets)
        
        if self.static_scaled is not None:
            self.static_tensor = torch.FloatTensor(self.static_scaled)
        else:
            self.static_tensor = None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.static_tensor is not None:
            return self.sequences_tensor[idx], self.static_tensor[idx], self.targets_tensor[idx]
        else:
            return self.sequences_tensor[idx], self.targets_tensor[idx]


class LSTMBookOrWaitModel(nn.Module):
    """LSTM model with static features for Book or Wait prediction."""
    
    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMBookOrWaitModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_input_size = static_input_size
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        fc_input_size = hidden_size + static_input_size
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, sequence_input, static_input=None):
        batch_size = sequence_input.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(sequence_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(sequence_input.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(sequence_input, (h0, c0))
        
        # Take the last output
        lstm_last = lstm_out[:, -1, :]
        
        # Combine with static features if available
        if static_input is not None and self.static_input_size > 0:
            combined = torch.cat((lstm_last, static_input), dim=1)
        else:
            combined = lstm_last
        
        # Fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x


def create_sequences(rental_prices_df, competitor_prices_df, sequence_length=14):
    """Create sequences of price data for LSTM training."""
    sequences = []
    targets = []
    static_features = []
    metadata = []
    
    # Get unique combinations
    price_keys = rental_prices_df[['supplier_id', 'location_id', 'car_class']].drop_duplicates()
    
    for _, key in price_keys.iterrows():
        supplier_id = key['supplier_id']
        location_id = key['location_id']
        car_class = key['car_class']
        
        # Get price history
        mask = (rental_prices_df['supplier_id'] == supplier_id) & \
               (rental_prices_df['location_id'] == location_id) & \
               (rental_prices_df['car_class'] == car_class)
        
        price_history = rental_prices_df[mask].sort_values('date')
        
        # Need enough data for sequences
        if len(price_history) < sequence_length + 7:
            continue
        
        # Create sequences
        for i in range(sequence_length, len(price_history) - 7):
            current_date = price_history.iloc[i]['date']
            current_price = price_history.iloc[i]['current_price']
            
            # Future prices for target
            future_prices = price_history.iloc[i+1:i+8]['current_price'].values
            max_future_price = np.max(future_prices)
            should_book = 1 if current_price < max_future_price else 0
            
            # Create sequence features (price history)
            sequence_data = []
            for j in range(i - sequence_length, i):
                row = price_history.iloc[j]
                # Sequence features: price, price changes, availability
                seq_features = [
                    row['current_price'],
                    row['available_cars'],
                    row['days_until_pickup']
                ]
                
                # Add price changes if we have enough history
                if j > 0:
                    prev_price = price_history.iloc[j-1]['current_price']
                    price_change = (row['current_price'] - prev_price) / prev_price
                    seq_features.append(price_change)
                else:
                    seq_features.append(0.0)
                
                sequence_data.append(seq_features)
            
            # Static features (unchanging characteristics)
            static_feature_vec = [
                supplier_id,
                location_id,
                current_date.dayofweek,
                current_date.month,
                1 if current_date.dayofweek >= 5 else 0,  # is_weekend
                1 if current_date.month in [6, 7, 8, 12] else 0,  # is_peak_season
                (current_date.month - 1) // 3 + 1,  # quarter
            ]
            
            # Add competitor features
            comp_mask = (competitor_prices_df['location_id'] == location_id) & \
                       (competitor_prices_df['car_class'] == car_class) & \
                       (competitor_prices_df['date'] == current_date)
            
            comp_prices = competitor_prices_df[comp_mask]['comp_min_price'].values
            if len(comp_prices) > 0:
                static_feature_vec.extend([
                    (current_price - np.mean(comp_prices)) / np.mean(comp_prices),  # price_vs_competitors
                    1 if current_price < np.min(comp_prices) else 0,  # is_cheapest
                    np.sum(current_price >= comp_prices) / len(comp_prices)  # price_rank
                ])
            else:
                static_feature_vec.extend([0, 0, 0.5])
            
            sequences.append(sequence_data)
            targets.append(should_book)
            static_features.append(static_feature_vec)
            
            # Store metadata for analysis
            metadata.append({
                'date': current_date,
                'supplier_id': supplier_id,
                'location_id': location_id,
                'car_class': car_class,
                'current_price': current_price
            })
    
    return (np.array(sequences), np.array(targets), 
            np.array(static_features), pd.DataFrame(metadata))


def train_epoch(model, train_loader, criterion, optimizer, device, has_static_features=True):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_data in train_loader:
        if has_static_features:
            batch_sequences, batch_static, batch_targets = batch_data
            batch_sequences = batch_sequences.to(device)
            batch_static = batch_static.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_sequences, batch_static).squeeze()
        else:
            batch_sequences, batch_targets = batch_data
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_sequences).squeeze()
        
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


def evaluate(model, test_loader, criterion, device, has_static_features=True):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if has_static_features:
                batch_sequences, batch_static, batch_targets = batch_data
                batch_sequences = batch_sequences.to(device)
                batch_static = batch_static.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                outputs = model(batch_sequences, batch_static).squeeze()
            else:
                batch_sequences, batch_targets = batch_data
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                outputs = model(batch_sequences).squeeze()
            
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
    
    # Create sequences
    print("\nCreating price sequences...")
    sequence_length = 14
    sequences, targets, static_features, metadata_df = create_sequences(
        rental_prices_df, competitor_prices_df, sequence_length
    )
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Static features shape: {static_features.shape}")
    print(f"Target distribution: {np.bincount(targets.astype(int)) / len(targets)}")
    
    # Time-based train/test split
    split_date = metadata_df['date'].quantile(0.8)
    train_mask = metadata_df['date'] < split_date
    test_mask = ~train_mask
    
    X_seq_train = sequences[train_mask]
    X_static_train = static_features[train_mask]
    y_train = targets[train_mask]
    
    X_seq_test = sequences[test_mask]
    X_static_test = static_features[test_mask]
    y_test = targets[test_mask]
    
    print(f"\nTrain sequences: {len(X_seq_train)}")
    print(f"Test sequences: {len(X_seq_test)}")
    
    # Create datasets
    train_dataset = PriceSequenceDataset(X_seq_train, y_train, X_static_train)
    test_dataset = PriceSequenceDataset(
        X_seq_test, y_test, X_static_test,
        scaler=train_dataset.scaler,
        sequence_scaler=train_dataset.sequence_scaler
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    sequence_input_size = sequences.shape[2]  # Number of features per timestep
    static_input_size = static_features.shape[1]  # Number of static features
    
    model = LSTMBookOrWaitModel(
        sequence_input_size=sequence_input_size,
        static_input_size=static_input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel initialized:")
    print(f"Sequence input size: {sequence_input_size}")
    print(f"Static input size: {static_input_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.001
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_auc = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc, _, _, test_probs = evaluate(model, test_loader, criterion, device)
        
        # Calculate AUC
        test_auc = roc_auc_score(y_test, test_probs)
        
        # Store history
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Save best model
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    _, _, predictions, targets_final, probabilities = evaluate(model, test_loader, criterion, device)
    
    print("\nClassification Report:")
    print(classification_report(targets_final, predictions, target_names=['Wait', 'Book Now']))
    
    # ROC-AUC Score
    final_auc = roc_auc_score(targets_final, probabilities)
    print(f"\nFinal ROC-AUC Score: {final_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(targets_final, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Wait', 'Book Now'], 
                yticklabels=['Wait', 'Book Now'])
    plt.title('LSTM Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'sequence_input_size': sequence_input_size,
            'static_input_size': static_input_size,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        },
        'sequence_length': sequence_length,
        'final_auc': final_auc,
        'train_losses': train_losses,
        'test_losses': test_losses
    }, model_dir / 'lstm_model.pth')
    
    # Save scalers
    joblib.dump(train_dataset.scaler, model_dir / 'lstm_static_scaler.pkl')
    joblib.dump(train_dataset.sequence_scaler, model_dir / 'lstm_sequence_scaler.pkl')
    
    print(f"\nModel saved to {model_dir}")
    print(f"Final ROC-AUC: {final_auc:.4f}")
    
    # Model comparison
    print(f"\nModel Comparison:")
    print(f"Logistic Regression AUC: 0.9032")
    print(f"XGBoost AUC: 0.9121")
    print(f"LSTM AUC: {final_auc:.4f}")


if __name__ == "__main__":
    main()
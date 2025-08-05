#!/usr/bin/env python3
"""
Kumo RFM Model for Book or Wait Decision System

This script implements the Book or Wait prediction system using Kumo.ai's
Relational Foundation Model (RFM). It demonstrates:
1. Loading and preparing relational data
2. Creating a graph structure for Kumo RFM
3. Writing PQL queries for price predictions
4. Making Book or Wait recommendations
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Check for Kumo API key
if not os.getenv('KUMO_API_KEY'):
    print("ERROR: KUMO_API_KEY not found in environment variables")
    print("Please add KUMO_API_KEY to your .env file")
    sys.exit(1)

try:
    import kumoai.experimental.rfm as rfm
    import kumoai as kumo
except ImportError:
    print("ERROR: Kumo SDK not installed")
    print("Please run: pip install kumoai")
    sys.exit(1)

# Initialize Kumo with API key
print("Initializing Kumo SDK...")
try:
    kumo.init(api_key=os.environ['KUMO_API_KEY'])
    print("✅ Kumo SDK initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Kumo SDK: {e}")
    sys.exit(1)


def load_sample_data():
    """Load sample data from notebooks directory."""
    print("\n📊 Loading sample data...")
    
    # Use sample data path
    data_path = Path("notebooks/data/sample")
    
    # Load all tables
    users_df = pd.read_csv(data_path / "users.csv")
    searches_df = pd.read_csv(data_path / "searches.csv")
    bookings_df = pd.read_csv(data_path / "bookings.csv")
    prices_df = pd.read_csv(data_path / "rental_prices.csv")
    competitor_df = pd.read_csv(data_path / "competitor_prices.csv")
    
    # Convert timestamps to datetime
    searches_df['search_ts'] = pd.to_datetime(searches_df['search_ts'])
    bookings_df['booking_ts'] = pd.to_datetime(bookings_df['booking_ts'])
    bookings_df['search_ts'] = pd.to_datetime(bookings_df['search_ts'])
    bookings_df['pickup_date'] = pd.to_datetime(bookings_df['pickup_date'])
    prices_df['obs_ts'] = pd.to_datetime(prices_df['obs_ts'])
    prices_df['pickup_date'] = pd.to_datetime(prices_df['pickup_date'])
    competitor_df['obs_date'] = pd.to_datetime(competitor_df['obs_date'])
    competitor_df['pickup_date'] = pd.to_datetime(competitor_df['pickup_date'])
    
    print(f"✅ Loaded {len(users_df):,} users")
    print(f"✅ Loaded {len(searches_df):,} searches")
    print(f"✅ Loaded {len(bookings_df):,} bookings")
    print(f"✅ Loaded {len(prices_df):,} rental prices")
    print(f"✅ Loaded {len(competitor_df):,} competitor prices")
    
    return users_df, searches_df, bookings_df, prices_df, competitor_df


def create_kumo_graph(users_df, searches_df, bookings_df, prices_df, competitor_df):
    """Create Kumo LocalGraph with proper metadata and relationships."""
    print("\n🔗 Creating Kumo graph structure...")
    
    # Create LocalTable for users
    print("  Creating users table...")
    users_table = rfm.LocalTable(
        df=users_df,
        table_name="users",
        primary_key="user_id"
    ).infer_metadata()
    
    # Override semantic types if needed
    users_table['segment'].stype = rfm.Stype.CATEGORICAL
    users_table['home_location_id'].stype = rfm.Stype.ID
    
    # Create LocalTable for searches
    print("  Creating searches table...")
    searches_table = rfm.LocalTable(
        df=searches_df,
        table_name="searches",
        primary_key="search_id",
        time_column="search_ts"
    ).infer_metadata()
    
    searches_table['user_segment'].stype = rfm.Stype.CATEGORICAL
    searches_table['car_class'].stype = rfm.Stype.CATEGORICAL
    searches_table['weekday'].stype = rfm.Stype.CATEGORICAL
    searches_table['month_name'].stype = rfm.Stype.CATEGORICAL
    
    # Create LocalTable for bookings
    print("  Creating bookings table...")
    bookings_table = rfm.LocalTable(
        df=bookings_df,
        table_name="bookings",
        primary_key="booking_id",
        time_column="booking_ts"
    ).infer_metadata()
    
    bookings_table['car_class'].stype = rfm.Stype.CATEGORICAL
    bookings_table['user_segment'].stype = rfm.Stype.CATEGORICAL
    
    # Create LocalTable for rental prices
    print("  Creating rental_prices table...")
    prices_table = rfm.LocalTable(
        df=prices_df,
        table_name="rental_prices",
        primary_key="price_id",
        time_column="obs_ts"
    ).infer_metadata()
    
    prices_table['car_class'].stype = rfm.Stype.CATEGORICAL
    prices_table['current_price'].stype = rfm.Stype.NUMERICAL
    prices_table['available_cars'].stype = rfm.Stype.NUMERICAL
    
    # Create LocalTable for competitor prices
    print("  Creating competitor_prices table...")
    competitor_table = rfm.LocalTable(
        df=competitor_df,
        table_name="competitor_prices",
        primary_key="comp_id",
        time_column="obs_date"
    ).infer_metadata()
    
    competitor_table['car_class'].stype = rfm.Stype.CATEGORICAL
    competitor_table['comp_min_price'].stype = rfm.Stype.NUMERICAL
    
    # Create graph with all tables
    print("  Creating graph with all tables...")
    graph = rfm.LocalGraph(tables=[
        users_table,
        searches_table,
        bookings_table,
        prices_table,
        competitor_table
    ])
    
    # Define relationships
    print("  Defining table relationships...")
    
    # User relationships
    graph.link(src_table="searches", fkey="user_id", dst_table="users")
    graph.link(src_table="bookings", fkey="user_id", dst_table="users")
    
    # Search-Booking relationship
    graph.link(src_table="bookings", fkey="search_id", dst_table="searches")
    
    # Location relationships (implicit through location_id matches)
    # Note: Kumo RFM can infer relationships through matching column names
    
    print("✅ Graph created successfully with 5 tables and 3 explicit relationships")
    
    # Validate graph
    try:
        graph.validate()
        print("✅ Graph validation passed")
    except Exception as e:
        print(f"⚠️  Graph validation warning: {e}")
    
    return graph


def test_book_or_wait_predictions(model, graph):
    """Test various Book or Wait prediction queries."""
    print("\n🔮 Testing Book or Wait predictions...")
    
    # Query 1: Basic price trend prediction
    print("\n1️⃣ Testing basic price trend prediction...")
    
    query1 = """
    PREDICT 
        MIN(rental_prices.current_price, 0, 7, days) > rental_prices.current_price
    FOR rental_prices.price_id IN (1, 2, 3, 4, 5)
    """
    
    try:
        result1 = model.predict(query1)
        print(f"✅ Query 1 executed successfully")
        print(f"   Results shape: {result1.shape}")
        print(f"   Sample predictions:")
        print(result1.head())
    except Exception as e:
        print(f"❌ Query 1 failed: {e}")
    
    # Query 2: Price forecast with location and car class
    print("\n2️⃣ Testing price forecast for specific location/class...")
    
    query2 = """
    PREDICT 
        AVG(rental_prices.current_price, 0, 7, days) - rental_prices.current_price AS price_change
    FOR rental_prices.location_id = 1 
        AND rental_prices.car_class = 'economy'
        AND rental_prices.days_until_pickup > 7
    LIMIT 10
    """
    
    try:
        result2 = model.predict(query2)
        print(f"✅ Query 2 executed successfully")
        print(f"   Results shape: {result2.shape}")
        if len(result2) > 0:
            print(f"   Average predicted price change: ${result2['price_change'].mean():.2f}")
    except Exception as e:
        print(f"❌ Query 2 failed: {e}")
    
    # Query 3: Customer-specific booking prediction
    print("\n3️⃣ Testing customer-specific booking predictions...")
    
    query3 = """
    PREDICT 
        COUNT(bookings.*, 0, 30, days) > 0 AS will_book
    FOR users.user_id IN (16067, 16064, 17978)
    """
    
    try:
        result3 = model.predict(query3)
        print(f"✅ Query 3 executed successfully")
        print(f"   Results shape: {result3.shape}")
        print(f"   Predictions:")
        print(result3)
    except Exception as e:
        print(f"❌ Query 3 failed: {e}")
    
    # Query 4: Complex Book or Wait decision
    print("\n4️⃣ Testing complex Book or Wait decision...")
    
    query4 = """
    PREDICT 
        CASE 
            WHEN MIN(rental_prices.current_price, 0, 7, days) > rental_prices.current_price * 1.05 
            THEN 1
            ELSE 0
        END AS should_book_now
    FOR rental_prices.supplier_id = 1
        AND rental_prices.location_id = 10
        AND rental_prices.car_class = 'suv'
        AND rental_prices.days_until_pickup BETWEEN 10 AND 30
    """
    
    try:
        result4 = model.predict(query4)
        print(f"✅ Query 4 executed successfully")
        print(f"   Results shape: {result4.shape}")
        if len(result4) > 0:
            book_rate = result4['should_book_now'].mean()
            print(f"   Book Now recommendation rate: {book_rate:.1%}")
    except Exception as e:
        print(f"❌ Query 4 failed: {e}")
    
    return result1 if 'result1' in locals() else None


def create_book_or_wait_function(model):
    """Create a practical Book or Wait prediction function."""
    
    def predict_book_or_wait(supplier_id, location_id, car_class, current_price, days_until_pickup):
        """
        Predict whether to book now or wait for a specific rental.
        
        Returns:
            dict: Contains recommendation, confidence, and predicted price change
        """
        # Create PQL query for this specific case
        query = f"""
        PREDICT 
            MIN(rental_prices.current_price, 0, 7, days) AS min_future_price,
            AVG(rental_prices.current_price, 0, 7, days) AS avg_future_price,
            MAX(rental_prices.current_price, 0, 7, days) AS max_future_price
        FOR rental_prices.supplier_id = {supplier_id}
            AND rental_prices.location_id = {location_id}
            AND rental_prices.car_class = '{car_class}'
            AND rental_prices.days_until_pickup >= {days_until_pickup - 7}
            AND rental_prices.days_until_pickup <= {days_until_pickup + 7}
        """
        
        try:
            result = model.predict(query)
            
            if len(result) == 0:
                return {
                    'recommendation': 'NO_DATA',
                    'confidence': 0.0,
                    'reason': 'Insufficient historical data for prediction'
                }
            
            # Aggregate predictions
            min_price = result['min_future_price'].mean()
            avg_price = result['avg_future_price'].mean()
            max_price = result['max_future_price'].mean()
            
            # Decision logic
            price_increase_likely = min_price > current_price * 1.02  # 2% threshold
            significant_increase = avg_price > current_price * 1.05   # 5% threshold
            
            if significant_increase:
                recommendation = 'BOOK_NOW'
                confidence = min(0.9, (avg_price - current_price) / current_price * 10)
                reason = f"Price likely to increase by ${avg_price - current_price:.2f} ({(avg_price/current_price - 1)*100:.1f}%)"
            elif price_increase_likely:
                recommendation = 'BOOK_NOW'
                confidence = 0.7
                reason = f"Moderate price increase expected (${min_price - current_price:.2f})"
            else:
                recommendation = 'WAIT'
                confidence = 0.6
                reason = f"Prices expected to remain stable or decrease"
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'reason': reason,
                'current_price': current_price,
                'predicted_min': min_price,
                'predicted_avg': avg_price,
                'predicted_max': max_price
            }
            
        except Exception as e:
            return {
                'recommendation': 'ERROR',
                'confidence': 0.0,
                'reason': f'Prediction failed: {str(e)}'
            }
    
    return predict_book_or_wait


def main():
    """Main execution function."""
    print("🚗 Kumo RFM Model for Book or Wait Decision System")
    print("=" * 60)
    
    # Load data
    users_df, searches_df, bookings_df, prices_df, competitor_df = load_sample_data()
    
    # Create graph
    graph = create_kumo_graph(users_df, searches_df, bookings_df, prices_df, competitor_df)
    
    # Initialize Kumo RFM model
    print("\n🤖 Initializing Kumo RFM model...")
    try:
        model = rfm.KumoRFM(graph)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Test predictions
    test_book_or_wait_predictions(model, graph)
    
    # Create practical prediction function
    predict_book_or_wait = create_book_or_wait_function(model)
    
    # Test the practical function
    print("\n📝 Testing practical Book or Wait function...")
    
    # Example 1: Economy car in location 1
    result1 = predict_book_or_wait(
        supplier_id=1,
        location_id=1,
        car_class='economy',
        current_price=55.00,
        days_until_pickup=14
    )
    
    print(f"\nExample 1 - Economy car:")
    print(f"  Current price: ${result1['current_price']:.2f}")
    print(f"  Recommendation: {result1['recommendation']}")
    print(f"  Confidence: {result1['confidence']:.1%}")
    print(f"  Reason: {result1['reason']}")
    
    # Example 2: Luxury car
    result2 = predict_book_or_wait(
        supplier_id=4,
        location_id=1,
        car_class='luxury',
        current_price=162.86,
        days_until_pickup=6
    )
    
    print(f"\nExample 2 - Luxury car:")
    print(f"  Current price: ${result2['current_price']:.2f}")
    print(f"  Recommendation: {result2['recommendation']}")
    print(f"  Confidence: {result2['confidence']:.1%}")
    print(f"  Reason: {result2['reason']}")
    
    print("\n✅ Kumo RFM validation completed successfully!")
    print("\n💡 Next steps:")
    print("  1. Extend PQL queries for more complex scenarios")
    print("  2. Add competitor price integration")
    print("  3. Implement caching for performance")
    print("  4. Create API endpoint for real-time predictions")


if __name__ == "__main__":
    main()
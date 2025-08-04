"""
Test script to validate the complete price-aware data generation pipeline.

This script tests:
1. Search generation with realistic patterns
2. Price generation with temporal variations
3. Booking generation using actual rental prices
4. Data integrity and relationships
"""

import pandas as pd
from price_aware_booking_generator import PriceAwareBookingGenerator
from realistic_price_generator import RealisticPriceGenerator

# Import our generators
from session_based_search_generator import SessionBasedSearchGenerator


def test_complete_pipeline():
    """Test the complete data generation pipeline."""
    print("=" * 70)
    print("TESTING COMPLETE PRICE-AWARE DATA GENERATION PIPELINE")
    print("=" * 70)

    # Parameters
    n_users = 1000
    n_locations = 24
    n_suppliers = 5
    car_classes = ["economy", "compact", "suv", "luxury"]

    # Date ranges
    search_start = "2024-06-01"
    search_end = "2024-06-30"
    price_start = "2024-05-01"  # Prices start earlier
    price_end = "2024-08-31"  # Prices extend later

    print("\nTest parameters:")
    print(f"  Users: {n_users:,}")
    print(f"  Locations: {n_locations}")
    print(f"  Suppliers: {n_suppliers}")
    print(f"  Car classes: {car_classes}")
    print(f"  Search period: {search_start} to {search_end}")

    # Step 1: Generate searches
    print("\n" + "-" * 50)
    print("STEP 1: Generating searches...")
    search_gen = SessionBasedSearchGenerator(
        n_users=n_users, start_date=search_start, end_date=search_end, seed=2025
    )
    searches_df = search_gen.generate_searches()
    print(f"✓ Generated {len(searches_df):,} searches")

    # Step 2: Generate rental prices
    print("\n" + "-" * 50)
    print("STEP 2: Generating rental prices...")
    price_gen = RealisticPriceGenerator(seed=2025)
    rental_prices_df = price_gen.generate_rental_prices(
        n_records=50000,
        start_date=price_start,
        end_date=price_end,
        locations=list(range(1, n_locations + 1)),
        suppliers=list(range(1, n_suppliers + 1)),
        car_classes=car_classes,
    )
    print(f"✓ Generated {len(rental_prices_df):,} rental price observations")

    # Step 3: Generate competitor prices
    print("\n" + "-" * 50)
    print("STEP 3: Generating competitor prices...")
    competitor_prices_df = price_gen.generate_competitor_prices(
        n_records=10000,
        start_date=price_start,
        end_date=price_end,
        locations=list(range(1, n_locations + 1)),
        car_classes=car_classes,
    )
    print(f"✓ Generated {len(competitor_prices_df):,} competitor price observations")

    # Step 4: Generate bookings using actual prices
    print("\n" + "-" * 50)
    print("STEP 4: Generating price-aware bookings...")
    booking_gen = PriceAwareBookingGenerator(seed=2025)
    bookings_df = booking_gen.generate_bookings_with_prices(
        searches_df=searches_df,
        rental_prices_df=rental_prices_df,
        competitor_prices_df=competitor_prices_df,
    )

    # Validation checks
    print("\n" + "=" * 50)
    print("VALIDATION CHECKS")
    print("=" * 50)

    # Check 1: Booking integrity
    print("\n1. Booking Integrity:")
    booking_users = set(bookings_df["user_id"])
    search_users = set(searches_df["user_id"])
    print(
        f"   ✓ All booking users exist in searches: {booking_users.issubset(search_users)}"
    )

    booking_searches = set(bookings_df["search_id"])
    all_searches = set(searches_df["search_id"])
    print(f"   ✓ All booking searches exist: {booking_searches.issubset(all_searches)}")

    # Check 2: Price validity
    print("\n2. Price Validity:")
    price_ranges = {
        "economy": (25, 150),
        "compact": (30, 180),
        "suv": (45, 250),
        "luxury": (75, 400),
    }

    for car_class in car_classes:
        class_bookings = bookings_df[bookings_df["car_class"] == car_class]
        if len(class_bookings) > 0:
            min_price = class_bookings["booked_price"].min()
            max_price = class_bookings["booked_price"].max()
            avg_price = class_bookings["booked_price"].mean()
            expected_range = price_ranges[car_class]
            valid = expected_range[0] <= min_price and max_price <= expected_range[1]
            print(
                f"   {car_class}: ${min_price:.2f}-${max_price:.2f} (avg: ${avg_price:.2f}) - Valid: {valid}"
            )

    # Check 3: Temporal consistency
    print("\n3. Temporal Consistency:")
    bookings_df["search_ts"] = pd.to_datetime(bookings_df["search_ts"])
    bookings_df["booking_ts"] = pd.to_datetime(bookings_df["booking_ts"])
    bookings_df["pickup_date"] = pd.to_datetime(bookings_df["pickup_date"])

    temporal_valid = (bookings_df["booking_ts"] >= bookings_df["search_ts"]).all() and (
        bookings_df["pickup_date"] > bookings_df["booking_ts"]
    ).all()
    print(f"   ✓ Search → Booking → Pickup order: {temporal_valid}")

    # Check 4: Supplier distribution
    print("\n4. Supplier Distribution:")
    supplier_counts = bookings_df["supplier_id"].value_counts().sort_index()
    supplier_names = {1: "Enterprise", 2: "Hertz", 3: "Avis", 4: "Budget", 5: "Sixt"}
    for supplier_id, count in supplier_counts.items():
        pct = count / len(bookings_df) * 100
        name = supplier_names.get(supplier_id, f"Supplier {supplier_id}")
        print(f"   {name}: {count:,} bookings ({pct:.1f}%)")

    # Check 5: Price competitiveness
    print("\n5. Price Competitiveness:")
    if "competitor_price" in bookings_df.columns:
        valid_comp = bookings_df.dropna(subset=["competitor_price"])
        if len(valid_comp) > 0:
            price_diff = valid_comp["booked_price"] - valid_comp["competitor_price"]
            print(f"   Average difference from competitor: ${price_diff.mean():.2f}")
            print(
                f"   Bookings below competitor price: {(price_diff < 0).sum() / len(valid_comp):.1%}"
            )
            print(
                f"   Bookings above competitor price: {(price_diff > 0).sum() / len(valid_comp):.1%}"
            )

    # Check 6: Conversion patterns by segment
    print("\n6. Conversion Rates by Segment:")
    if "user_segment" in searches_df.columns:
        for segment in searches_df["user_segment"].unique():
            segment_searches = searches_df[searches_df["user_segment"] == segment]
            segment_bookings = (
                bookings_df[bookings_df["user_segment"] == segment]
                if "user_segment" in bookings_df.columns
                else pd.DataFrame()
            )

            conversion_rate = (
                len(segment_bookings) / len(segment_searches)
                if len(segment_searches) > 0
                else 0
            )
            print(
                f"   {segment}: {conversion_rate:.1%} ({len(segment_bookings):,}/{len(segment_searches):,})"
            )

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"\nTotal searches: {len(searches_df):,}")
    print(f"Total bookings: {len(bookings_df):,}")
    print(f"Overall conversion rate: {len(bookings_df) / len(searches_df):.2%}")
    print(f"Unique users who searched: {searches_df['user_id'].nunique():,}")
    print(f"Unique users who booked: {bookings_df['user_id'].nunique():,}")
    print(f"Average searches per user: {len(searches_df) / n_users:.2f}")
    print(f"Average bookings per user: {len(bookings_df) / n_users:.2f}")

    # Save test results
    print("\n" + "-" * 50)
    print("Saving test data...")

    # Create test output directory
    import os

    os.makedirs("data/test_pipeline", exist_ok=True)

    # Save data
    searches_df.to_csv("data/test_pipeline/searches.csv", index=False)
    rental_prices_df.to_csv("data/test_pipeline/rental_prices.csv", index=False)
    competitor_prices_df.to_csv("data/test_pipeline/competitor_prices.csv", index=False)
    bookings_df.to_csv("data/test_pipeline/bookings.csv", index=False)

    print("\n✅ Test data saved to data/test_pipeline/")
    print("   - searches.csv")
    print("   - rental_prices.csv")
    print("   - competitor_prices.csv")
    print("   - bookings.csv")

    return {
        "searches": searches_df,
        "rental_prices": rental_prices_df,
        "competitor_prices": competitor_prices_df,
        "bookings": bookings_df,
    }


if __name__ == "__main__":
    # Run the test
    results = test_complete_pipeline()

    print("\n" + "=" * 70)
    print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

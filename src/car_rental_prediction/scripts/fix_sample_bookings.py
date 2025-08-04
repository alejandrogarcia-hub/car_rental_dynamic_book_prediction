"""
Script to demonstrate the proper way to generate bookings linked to users and searches.
This shows how to fix the issue in notebooks/01_sample_dataset_generator.ipynb
"""

import numpy as np
import pandas as pd

# Set random seed
rng = np.random.default_rng(2025)

# Load or create sample data
print("Creating sample data...")

# Sample searches (from notebook)
n_searches = 200_000
n_users = 20_000

# Create searches with user assignments
searches_df = pd.DataFrame(
    {
        "search_id": range(1, n_searches + 1),
        "user_id": rng.integers(1, n_users + 1, n_searches),
        "location_id": rng.integers(1, 25, n_searches),
        "car_class": rng.choice(["economy", "compact", "suv", "luxury"], n_searches),
        "search_ts": pd.to_datetime("2024-07-24")
        + pd.to_timedelta(rng.integers(-365 * 24, 0, n_searches), unit="h"),
    }
)

print(
    f"Created {len(searches_df):,} searches from {searches_df['user_id'].nunique():,} unique users"
)

# OLD WAY (WRONG - doesn't link to users properly)
print("\n" + "=" * 60)
print("OLD WAY - Random bookings without user linkage:")
print("=" * 60)

n_bookings_old = int(searches_df.shape[0] * 0.025)
bookings_old = pd.DataFrame(
    {
        "booking_id": range(1, n_bookings_old + 1),
        "search_id": rng.integers(1, n_searches + 1, n_bookings_old),
        "supplier_id": rng.integers(1, 6, n_bookings_old),
        "supplier_location_id": rng.integers(1, 25, n_bookings_old),
        "booking_ts": pd.to_datetime("2024-07-24")
        + pd.to_timedelta(rng.integers(-365 * 24, 0, n_bookings_old), unit="h"),
        "booked_price": rng.normal(68, 18, n_bookings_old).clip(25, 200),
    }
)

# Check issues with old way
old_search_ids = set(bookings_old["search_id"])
print(f"  Generated {len(bookings_old):,} bookings")
print("  ❌ No user_id column in bookings")
print("  ❌ No car_class information from search")
print("  ❌ No guarantee booking happens after search")

# NEW WAY (CORRECT - properly linked)
print("\n" + "=" * 60)
print("NEW WAY - Properly linked bookings:")
print("=" * 60)

# Step 1: Select which searches convert to bookings (2.5% conversion rate)
conversion_rate = 0.025
booking_mask = rng.random(len(searches_df)) < conversion_rate
booking_searches = searches_df[booking_mask].copy()

print(f"  Selected {len(booking_searches):,} searches that convert to bookings")
print(f"  Conversion rate: {len(booking_searches) / len(searches_df):.2%}")

# Step 2: Generate bookings from these searches
bookings_new = []
for idx, (_, search) in enumerate(booking_searches.iterrows(), 1):
    # Booking happens 0.5 to 48 hours after search
    booking_delay_hours = rng.uniform(0.5, 48)
    booking_ts = search["search_ts"] + pd.Timedelta(hours=booking_delay_hours)

    # Price depends on car class
    price_factors = {"economy": 1.0, "compact": 1.15, "suv": 1.45, "luxury": 2.2}
    base_price = rng.normal(65, 15)
    price = base_price * price_factors.get(search["car_class"], 1.0)
    price = np.clip(price, 25, 250)

    bookings_new.append(
        {
            "booking_id": idx,
            "search_id": search["search_id"],
            "user_id": search["user_id"],  # Link to user!
            "supplier_id": rng.integers(1, 6),
            "location_id": search["location_id"],  # Same location as search
            "car_class": search["car_class"],  # Include car class!
            "search_ts": search["search_ts"],
            "booking_ts": booking_ts,
            "booked_price": price,
        }
    )

bookings_new_df = pd.DataFrame(bookings_new)

print("\n  ✅ All bookings have user_id from searches")
print("  ✅ All bookings include car_class from searches")
print("  ✅ All bookings happen after their search")
print("  ✅ Prices vary by car class")

# Validate the new approach
print("\n" + "=" * 60)
print("VALIDATION:")
print("=" * 60)

# Check 1: User linkage
unique_booking_users = bookings_new_df["user_id"].nunique()
print(f"  Users who made bookings: {unique_booking_users:,}")

# Check 2: Car class distribution
print("\n  Bookings by car class:")
for car_class, count in bookings_new_df["car_class"].value_counts().items():
    avg_price = bookings_new_df[bookings_new_df["car_class"] == car_class][
        "booked_price"
    ].mean()
    print(f"    {car_class}: {count:,} bookings, avg price: ${avg_price:.2f}")

# Check 3: Temporal consistency
time_travel = bookings_new_df["booking_ts"] < bookings_new_df["search_ts"]
print(f"\n  Bookings before search: {time_travel.sum()} (should be 0)")

# Check 4: User booking frequency
user_booking_counts = bookings_new_df["user_id"].value_counts()
print("\n  Users with multiple bookings:")
print(f"    1 booking: {(user_booking_counts == 1).sum():,} users")
print(f"    2 bookings: {(user_booking_counts == 2).sum():,} users")
print(f"    3+ bookings: {(user_booking_counts >= 3).sum():,} users")

# Save corrected data
print("\n" + "=" * 60)
print("RECOMMENDED NOTEBOOK UPDATE:")
print("=" * 60)

print("""
Replace the bookings generation code in the notebook with:

# Generate bookings (2.5% conversion rate)
conversion_rate = 0.025
booking_mask = rng.random(len(searches_df)) < conversion_rate
booking_searches = searches_df[booking_mask].copy()

bookings = []
for idx, (_, search) in enumerate(booking_searches.iterrows(), 1):
    # Booking happens 0.5 to 48 hours after search
    booking_delay_hours = rng.uniform(0.5, 48)
    booking_ts = search["search_ts"] + pd.Timedelta(hours=booking_delay_hours)
    
    # Price depends on car class
    price_factors = {"economy": 1.0, "compact": 1.15, "suv": 1.45, "luxury": 2.2}
    base_price = rng.normal(65, 15)
    price = base_price * price_factors.get(search["car_class"], 1.0)
    price = np.clip(price, 25, 250)
    
    bookings.append({
        "booking_id": idx,
        "search_id": search["search_id"],
        "user_id": search["user_id"],
        "supplier_id": rng.integers(1, 6),
        "location_id": search["location_id"],
        "car_class": search["car_class"],
        "search_ts": search["search_ts"],
        "booking_ts": booking_ts,
        "booked_price": price,
    })

bookings_df = pd.DataFrame(bookings)
print(f"Generated {len(bookings_df):,} bookings from {len(searches_df):,} searches")
print(f"Conversion rate: {len(bookings_df)/len(searches_df):.2%}")
""")

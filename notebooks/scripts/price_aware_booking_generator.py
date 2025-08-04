"""
Enhanced booking generator that uses actual rental prices from price data.

This module extends the realistic booking generator to:
- Look up actual prices from rental_prices data
- Consider price in booking decisions
- Track price at search time vs booking time
- Implement price-sensitive conversion rates
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


class PriceAwareBookingGenerator:
    """
    Generates bookings using actual rental price data.

    This generator:
    - Matches searches with available rental prices
    - Implements price-sensitive conversion rates
    - Tracks price changes between search and booking
    - Ensures booked prices come from actual price data
    """

    def __init__(self, seed: int = 2025):
        """Initialize the price-aware booking generator."""
        self.rng = np.random.default_rng(seed)

        # Base conversion rates by segment
        self.segment_conversion_rates = {
            "non_searcher": 0.0,
            "browser_only": 0.0,
            "single_trip": 0.175,
            "multi_trip": 0.45,
            "frequent_renter": 0.65,
        }

        # Price sensitivity factors (how much price affects conversion)
        # Higher value = more price sensitive
        self.price_sensitivity = {
            "economy": 1.5,  # Most price sensitive
            "compact": 1.2,
            "suv": 0.8,
            "luxury": 0.5,  # Least price sensitive
        }

        # Expected price ranges by car class (for conversion calculations)
        self.expected_prices = {
            "economy": {"min": 35, "typical": 50, "max": 80},
            "compact": {"min": 40, "typical": 58, "max": 95},
            "suv": {"min": 60, "typical": 85, "max": 140},
            "luxury": {"min": 100, "typical": 140, "max": 250},
        }

    def generate_bookings_with_prices(
        self,
        searches_df: pd.DataFrame,
        rental_prices_df: pd.DataFrame,
        competitor_prices_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate bookings using actual rental price data.

        Args:
            searches_df: Search data with columns: search_id, user_id, location_id,
                        car_class, search_ts, user_segment
            rental_prices_df: Rental price data with columns: location_id, supplier_id,
                             car_class, pickup_date, obs_ts, current_price
            competitor_prices_df: Optional competitor price data

        Returns:
            DataFrame with bookings including actual prices from rental_prices
        """
        print("Generating price-aware bookings...")

        # Validate inputs
        self._validate_inputs(searches_df, rental_prices_df)

        # Ensure datetime columns
        searches_df = searches_df.copy()
        rental_prices_df = rental_prices_df.copy()
        searches_df["search_ts"] = pd.to_datetime(searches_df["search_ts"])
        rental_prices_df["obs_ts"] = pd.to_datetime(rental_prices_df["obs_ts"])
        rental_prices_df["pickup_date"] = pd.to_datetime(
            rental_prices_df["pickup_date"]
        )

        # Process each search
        bookings = []
        no_price_found = 0

        for _, search in searches_df.iterrows():
            # Find available prices for this search
            available_prices = self._find_available_prices(
                search, rental_prices_df, competitor_prices_df
            )

            if available_prices.empty:
                no_price_found += 1
                continue

            # Decide if this search converts to booking
            booking = self._attempt_booking(search, available_prices)
            if booking is not None:
                bookings.append(booking)

        if no_price_found > 0:
            print(f"Warning: {no_price_found} searches had no matching prices")

        # Create bookings DataFrame
        if bookings:
            bookings_df = pd.DataFrame(bookings)
            bookings_df = self._finalize_bookings(bookings_df, searches_df)

            print(
                f"Generated {len(bookings_df):,} bookings from {len(searches_df):,} searches"
            )
            print(f"Conversion rate: {len(bookings_df) / len(searches_df):.2%}")

            self._print_booking_summary(bookings_df)
            return bookings_df
        else:
            print("No bookings generated")
            return pd.DataFrame()

    def _validate_inputs(
        self, searches_df: pd.DataFrame, rental_prices_df: pd.DataFrame
    ) -> None:
        """Validate required columns exist."""
        search_required = [
            "search_id",
            "user_id",
            "location_id",
            "car_class",
            "search_ts",
        ]
        price_required = [
            "location_id",
            "supplier_id",
            "car_class",
            "pickup_date",
            "obs_ts",
            "current_price",
        ]

        missing_search = set(search_required) - set(searches_df.columns)
        missing_price = set(price_required) - set(rental_prices_df.columns)

        if missing_search:
            raise ValueError(f"Missing columns in searches_df: {missing_search}")
        if missing_price:
            raise ValueError(f"Missing columns in rental_prices_df: {missing_price}")

    def _find_available_prices(
        self,
        search: pd.Series,
        rental_prices_df: pd.DataFrame,
        competitor_prices_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Find rental prices available at the time of search."""
        # Prices must be:
        # 1. Observed before or at search time
        # 2. For pickup dates after search time
        # 3. Matching location and car class

        search_time = search["search_ts"]

        # Filter rental prices
        mask = (
            (rental_prices_df["obs_ts"] <= search_time)
            & (rental_prices_df["pickup_date"] > search_time)
            & (rental_prices_df["location_id"] == search["location_id"])
            & (rental_prices_df["car_class"] == search["car_class"])
            & (rental_prices_df["available_cars"] > 0)  # Must have availability
        )

        available = rental_prices_df[mask].copy()

        if available.empty:
            return available

        # For each supplier/pickup_date combo, get the most recent observation
        available = (
            available.sort_values("obs_ts")
            .groupby(["supplier_id", "pickup_date"])
            .last()
            .reset_index()
        )

        # Add competitor price if available
        if competitor_prices_df is not None:
            comp_price = self._get_competitor_price(search, competitor_prices_df)
            if comp_price is not None:
                available["competitor_price"] = comp_price
            else:
                available["competitor_price"] = available["current_price"].min()

        # Calculate days until pickup
        available["days_until_pickup"] = (
            available["pickup_date"] - search_time
        ).dt.total_seconds() / 86400

        # Filter to reasonable booking window (1-60 days)
        available = available[
            (available["days_until_pickup"] >= 1)
            & (available["days_until_pickup"] <= 60)
        ]

        return available

    def _get_competitor_price(
        self, search: pd.Series, competitor_prices_df: pd.DataFrame
    ) -> float | None:
        """Get competitor price for comparison."""
        search_date = pd.Timestamp(search["search_ts"].date())

        # Ensure obs_date is datetime
        competitor_prices_df["obs_date"] = pd.to_datetime(
            competitor_prices_df["obs_date"]
        )

        mask = (
            (competitor_prices_df["obs_date"] <= search_date)
            & (competitor_prices_df["location_id"] == search["location_id"])
            & (competitor_prices_df["car_class"] == search["car_class"])
        )

        matches = competitor_prices_df[mask]
        if matches.empty:
            return None

        # Get most recent competitor price
        return matches.sort_values("obs_date").iloc[-1]["comp_min_price"]

    def _attempt_booking(
        self, search: pd.Series, available_prices: pd.DataFrame
    ) -> dict | None:
        """Decide if search converts to booking based on prices."""
        # Get base conversion rate
        segment = search.get("user_segment", "single_trip")
        base_rate = self.segment_conversion_rates.get(segment, 0.025)

        if base_rate == 0:
            return None

        # Find best price option
        best_option = available_prices.loc[available_prices["current_price"].idxmin()]
        price = best_option["current_price"]

        # Calculate price-adjusted conversion rate
        car_class = search["car_class"]
        expected = self.expected_prices[car_class]["typical"]
        price_ratio = price / expected

        # Adjust conversion rate based on price
        # If price is at expected level, no adjustment
        # If price is 50% higher, reduce conversion by sensitivity factor
        sensitivity = self.price_sensitivity[car_class]
        price_adjustment = 1.0 - sensitivity * max(0, price_ratio - 1.0)
        price_adjustment = max(0.1, min(2.0, price_adjustment))  # Bounds

        final_rate = base_rate * price_adjustment

        # Random decision
        if self.rng.random() >= final_rate:
            return None

        # Create booking
        booking_delay_hours = self._generate_booking_delay()
        booking_ts = search["search_ts"] + timedelta(hours=booking_delay_hours)

        # Ensure booking happens before pickup
        max_booking_time = best_option["pickup_date"] - timedelta(hours=4)
        if booking_ts > max_booking_time:
            booking_ts = max_booking_time
            booking_delay_hours = (
                booking_ts - search["search_ts"]
            ).total_seconds() / 3600

        return {
            "search_id": search["search_id"],
            "user_id": search["user_id"],
            "supplier_id": best_option["supplier_id"],
            "location_id": search["location_id"],
            "car_class": search["car_class"],
            "pickup_date": best_option["pickup_date"],
            "search_ts": search["search_ts"],
            "booking_ts": booking_ts,
            "booking_delay_hours": booking_delay_hours,
            "search_price": price,  # Price at search time
            "booked_price": price,  # Could be different in reality
            "price_rank": 1,  # Best price = 1
            "competitor_price": best_option.get("competitor_price", np.nan),
            "days_until_pickup": best_option["days_until_pickup"],
        }

    def _generate_booking_delay(self) -> float:
        """Generate realistic delay between search and booking."""
        rand = self.rng.random()

        if rand < 0.4:
            # Within 1 hour
            return self.rng.exponential(0.3)
        elif rand < 0.7:
            # 1-24 hours
            return self.rng.uniform(1, 24)
        elif rand < 0.9:
            # 1-7 days
            return self.rng.uniform(24, 168)
        else:
            # 7+ days
            return 168 + self.rng.exponential(120)

    def _finalize_bookings(
        self, bookings_df: pd.DataFrame, searches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Finalize booking data with proper IDs and sorting."""
        # Sort by booking timestamp
        bookings_df = bookings_df.sort_values("booking_ts").reset_index(drop=True)

        # Add booking IDs
        bookings_df["booking_id"] = range(1, len(bookings_df) + 1)

        # Add user segment if available
        if "user_segment" in searches_df.columns:
            segment_map = searches_df.set_index("search_id")["user_segment"].to_dict()
            bookings_df["user_segment"] = bookings_df["search_id"].map(segment_map)

        # Reorder columns
        column_order = [
            "booking_id",
            "search_id",
            "user_id",
            "supplier_id",
            "location_id",
            "car_class",
            "pickup_date",
            "search_ts",
            "booking_ts",
            "booking_delay_hours",
            "days_until_pickup",
            "search_price",
            "booked_price",
            "competitor_price",
            "price_rank",
        ]

        # Add user_segment if present
        if "user_segment" in bookings_df.columns:
            column_order.append("user_segment")

        return bookings_df[column_order]

    def _print_booking_summary(self, bookings_df: pd.DataFrame) -> None:
        """Print comprehensive booking summary."""
        print("\n" + "=" * 60)
        print("PRICE-AWARE BOOKING SUMMARY")
        print("=" * 60)

        print(f"\nTotal bookings: {len(bookings_df):,}")

        # Price statistics
        print("\nPrice Statistics:")
        print(f"  Average booked price: ${bookings_df['booked_price'].mean():.2f}")
        print(
            f"  Price range: ${bookings_df['booked_price'].min():.2f} - ${bookings_df['booked_price'].max():.2f}"
        )

        # Price by car class
        print("\nAverage price by car class:")
        for car_class in bookings_df["car_class"].unique():
            class_data = bookings_df[bookings_df["car_class"] == car_class]
            avg_price = class_data["booked_price"].mean()
            count = len(class_data)
            print(f"  {car_class}: ${avg_price:.2f} ({count:,} bookings)")

        # Supplier distribution
        print("\nBookings by supplier:")
        supplier_counts = bookings_df["supplier_id"].value_counts().sort_index()
        for supplier_id, count in supplier_counts.items():
            pct = count / len(bookings_df) * 100
            avg_price = bookings_df[bookings_df["supplier_id"] == supplier_id][
                "booked_price"
            ].mean()
            print(
                f"  Supplier {supplier_id}: {count:,} ({pct:.1f}%) - Avg: ${avg_price:.2f}"
            )

        # Booking timing
        print("\nBooking timing:")
        print(
            f"  Avg days before pickup: {bookings_df['days_until_pickup'].mean():.1f}"
        )
        print(
            f"  Booking delay < 1 hour: {(bookings_df['booking_delay_hours'] < 1).sum() / len(bookings_df):.1%}"
        )
        print(
            f"  Booking delay < 24 hours: {(bookings_df['booking_delay_hours'] < 24).sum() / len(bookings_df):.1%}"
        )

        # Price competitiveness
        if (
            "competitor_price" in bookings_df.columns
            and not bookings_df["competitor_price"].isna().all()
        ):
            price_diff = bookings_df["booked_price"] - bookings_df["competitor_price"]
            print("\nPrice competitiveness:")
            print(f"  Avg difference from competitor: ${price_diff.mean():.2f}")
            print(
                f"  Booked below competitor price: {(price_diff < 0).sum() / len(bookings_df):.1%}"
            )


# Example usage
if __name__ == "__main__":
    # Generate sample data for testing
    print("Generating sample data...")

    # Sample searches
    n_searches = 10000
    rng = np.random.default_rng(2025)

    searches_df = pd.DataFrame(
        {
            "search_id": range(1, n_searches + 1),
            "user_id": rng.integers(1, 1000, n_searches),
            "location_id": rng.integers(1, 25, n_searches),
            "car_class": rng.choice(
                ["economy", "compact", "suv", "luxury"], n_searches
            ),
            "search_ts": pd.to_datetime("2024-06-01")
            + pd.to_timedelta(rng.integers(0, 30 * 24, n_searches), unit="h"),
            "user_segment": rng.choice(
                ["single_trip", "multi_trip", "frequent_renter"],
                n_searches,
                p=[0.6, 0.3, 0.1],
            ),
        }
    )

    # Generate rental prices
    from realistic_price_generator import RealisticPriceGenerator

    price_gen = RealisticPriceGenerator()
    rental_prices_df = price_gen.generate_rental_prices(
        n_records=50000,
        start_date="2024-05-01",
        end_date="2024-07-31",
        locations=list(range(1, 25)),
        suppliers=[1, 2, 3, 4, 5],
        car_classes=["economy", "compact", "suv", "luxury"],
    )

    # Generate bookings
    booking_gen = PriceAwareBookingGenerator()
    bookings_df = booking_gen.generate_bookings_with_prices(
        searches_df, rental_prices_df
    )

    # Save results
    if not bookings_df.empty:
        bookings_df.to_csv("data/price_aware_bookings.csv", index=False)
        print("\n✅ Bookings saved to: data/price_aware_bookings.csv")
    else:
        print("\n❌ No bookings generated")

"""
Realistic booking generator that creates bookings based on user searches.

This module generates synthetic booking data that follows real-world patterns:
- 2.5% conversion rate from searches to bookings
- Bookings linked to actual users and their searches
- Temporal consistency (bookings occur after searches)
- Price variations based on car class and timing
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


class RealisticBookingGenerator:
    """
    Generates realistic booking data based on search patterns.

    Key features:
    - Links bookings to users who made searches
    - Respects 2.5% conversion rate
    - Includes car class from search
    - Realistic price distributions by car class
    - Temporal consistency with searches
    """

    def __init__(self, seed: int = 2025):
        """
        Initialize the booking generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

        # Conversion rates by user segment (from research)
        self.segment_conversion_rates = {
            "non_searcher": 0.0,  # Never book (by definition)
            "browser_only": 0.0,  # Never book online
            "single_trip": 0.175,  # 17.5% conversion
            "multi_trip": 0.45,  # 45% conversion
            "frequent_renter": 0.65,  # 65% conversion
        }

        # Base conversion rate for users without segments
        self.default_conversion_rate = 0.025  # 2.5% overall

        # Price multipliers by car class
        self.car_class_price_factors = {
            "economy": 1.0,  # Base price
            "compact": 1.15,  # 15% more than economy
            "suv": 1.45,  # 45% more than economy
            "luxury": 2.2,  # 120% more than economy
        }

        # Base price parameters (economy car)
        self.base_price_mean = 65
        self.base_price_std = 15

    def generate_bookings(
        self,
        searches_df: pd.DataFrame,
        users_df: pd.DataFrame | None = None,
        target_conversion_rate: float | None = None,
    ) -> pd.DataFrame:
        """
        Generate bookings based on search data.

        Args:
            searches_df: DataFrame with search data (must include user_id, search_id, car_class, search_ts)
            users_df: Optional DataFrame with user data (for segment-based conversion)
            target_conversion_rate: Override default conversion rate

        Returns:
            DataFrame with booking records
        """
        # Validate required columns
        required_cols = [
            "search_id",
            "user_id",
            "car_class",
            "search_ts",
            "location_id",
        ]
        missing_cols = set(required_cols) - set(searches_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in searches_df: {missing_cols}")

        # Sort searches by timestamp to ensure temporal consistency
        searches_df = searches_df.sort_values("search_ts").copy()

        # Determine which searches convert to bookings
        if users_df is not None and "user_segment" in searches_df.columns:
            # Use segment-based conversion rates
            booking_mask = self._apply_segment_conversion(searches_df)
        else:
            # Use default conversion rate
            conversion_rate = target_conversion_rate or self.default_conversion_rate
            booking_mask = self.rng.random(len(searches_df)) < conversion_rate

        # Select searches that convert to bookings
        booking_searches = searches_df[booking_mask].copy()

        if len(booking_searches) == 0:
            print(
                "Warning: No bookings generated (conversion rate too low or no searches)"
            )
            return pd.DataFrame()

        print(
            f"Generated {len(booking_searches):,} bookings from {len(searches_df):,} searches"
        )
        print(f"Actual conversion rate: {len(booking_searches) / len(searches_df):.2%}")

        # Generate booking details
        bookings = []
        for idx, (_, search) in enumerate(booking_searches.iterrows(), 1):
            booking = self._create_booking(idx, search)
            bookings.append(booking)

        bookings_df = pd.DataFrame(bookings)

        # Validate and summarize
        self._validate_bookings(bookings_df, searches_df)
        self._print_booking_summary(bookings_df)

        return bookings_df

    def _apply_segment_conversion(self, searches_df: pd.DataFrame) -> np.ndarray:
        """Apply segment-specific conversion rates."""
        booking_mask = np.zeros(len(searches_df), dtype=bool)

        for segment, rate in self.segment_conversion_rates.items():
            if rate > 0:  # Skip segments that never convert
                segment_mask = searches_df["user_segment"] == segment
                segment_indices = np.where(segment_mask)[0]

                # Apply conversion rate to this segment
                n_conversions = int(len(segment_indices) * rate)
                if n_conversions > 0:
                    # Randomly select which searches convert
                    converting_indices = self.rng.choice(
                        segment_indices, size=n_conversions, replace=False
                    )
                    booking_mask[converting_indices] = True

        return booking_mask

    def _create_booking(self, booking_id: int, search: pd.Series) -> dict:
        """Create a single booking from a search."""
        # Time between search and booking (minutes to days)
        # Most bookings happen within hours, some take days
        booking_delay_hours = self._generate_booking_delay()
        booking_ts = search["search_ts"] + timedelta(hours=booking_delay_hours)

        # Generate price based on car class
        booked_price = self._generate_price(search["car_class"])

        # Select supplier (could be enhanced with preference patterns)
        supplier_id = self._select_supplier(search.get("location_id", 1))

        return {
            "booking_id": booking_id,
            "search_id": search["search_id"],
            "user_id": search["user_id"],
            "supplier_id": supplier_id,
            "location_id": search["location_id"],
            "car_class": search["car_class"],
            "search_ts": search["search_ts"],
            "booking_ts": booking_ts,
            "booking_delay_hours": booking_delay_hours,
            "booked_price": booked_price,
        }

    def _generate_booking_delay(self) -> float:
        """
        Generate realistic delay between search and booking.

        Research shows:
        - 40% book within 1 hour
        - 30% book within 1-24 hours
        - 20% book within 1-7 days
        - 10% book after 7 days
        """
        rand = self.rng.random()

        if rand < 0.4:
            # Within 1 hour (exponential distribution)
            return self.rng.exponential(0.3)  # Mean ~20 minutes
        elif rand < 0.7:
            # 1-24 hours (uniform)
            return self.rng.uniform(1, 24)
        elif rand < 0.9:
            # 1-7 days (uniform)
            return self.rng.uniform(24, 168)
        else:
            # 7-30 days (exponential decay)
            return 168 + self.rng.exponential(120)  # Mean ~5 days after week 1

    def _generate_price(self, car_class: str) -> float:
        """Generate price based on car class."""
        # Get price factor for car class
        price_factor = self.car_class_price_factors.get(car_class, 1.0)

        # Generate base price with some variation
        base_price = self.rng.normal(self.base_price_mean, self.base_price_std)

        # Apply car class factor
        price = base_price * price_factor

        # Add some random variation (±10%)
        price *= self.rng.uniform(0.9, 1.1)

        # Ensure reasonable bounds
        min_price = 25 * price_factor
        max_price = 250 * price_factor

        return np.clip(price, min_price, max_price)

    def _select_supplier(self, location_id: int) -> int:  # noqa: ARG002
        """
        Select supplier for booking.

        In reality, this would depend on:
        - Price competitiveness
        - Availability
        - User preferences/loyalty
        - Supplier market share

        For now, using weighted random selection.
        """
        # Supplier market share weights (based on US market)
        supplier_weights = {
            1: 0.30,  # Enterprise (largest)
            2: 0.22,  # Hertz
            3: 0.20,  # Avis
            4: 0.18,  # Budget
            5: 0.10,  # Others (Sixt, etc.)
        }

        suppliers = list(supplier_weights.keys())
        weights = list(supplier_weights.values())

        return self.rng.choice(suppliers, p=weights)

    def _validate_bookings(
        self, bookings_df: pd.DataFrame, searches_df: pd.DataFrame
    ) -> None:
        """Validate booking data integrity."""
        # Check all bookings have valid search IDs
        valid_search_ids = set(searches_df["search_id"])
        invalid_bookings = ~bookings_df["search_id"].isin(valid_search_ids)
        if invalid_bookings.any():
            raise ValueError(
                f"Found {invalid_bookings.sum()} bookings with invalid search IDs"
            )

        # Check all bookings have valid user IDs
        valid_user_ids = set(searches_df["user_id"])
        invalid_users = ~bookings_df["user_id"].isin(valid_user_ids)
        if invalid_users.any():
            raise ValueError(
                f"Found {invalid_users.sum()} bookings with invalid user IDs"
            )

        # Check temporal consistency
        merged = bookings_df.merge(
            searches_df[["search_id", "search_ts"]],
            on="search_id",
            suffixes=("", "_search"),
        )

        time_travel = merged["booking_ts"] < merged["search_ts_search"]
        if time_travel.any():
            raise ValueError(
                f"Found {time_travel.sum()} bookings that occur before their search"
            )

        print("✅ All booking validations passed")

    def _print_booking_summary(self, bookings_df: pd.DataFrame) -> None:
        """Print summary statistics about generated bookings."""
        print("\nBooking Summary:")
        print(f"Total bookings: {len(bookings_df):,}")

        # Bookings by car class
        print("\nBookings by car class:")
        car_class_dist = bookings_df["car_class"].value_counts()
        for car_class, count in car_class_dist.items():
            pct = count / len(bookings_df) * 100
            avg_price = bookings_df[bookings_df["car_class"] == car_class][
                "booked_price"
            ].mean()
            print(
                f"  {car_class}: {count:,} ({pct:.1f}%) - Avg price: ${avg_price:.2f}"
            )

        # Booking delay statistics
        print("\nBooking delay statistics:")
        delays = bookings_df["booking_delay_hours"]
        print(f"  Median delay: {delays.median():.1f} hours")
        print(f"  Within 1 hour: {(delays <= 1).sum() / len(delays):.1%}")
        print(f"  Within 24 hours: {(delays <= 24).sum() / len(delays):.1%}")
        print(f"  Within 7 days: {(delays <= 168).sum() / len(delays):.1%}")

        # Price statistics
        print("\nPrice statistics:")
        print(f"  Overall average: ${bookings_df['booked_price'].mean():.2f}")
        print(
            f"  Price range: ${bookings_df['booked_price'].min():.2f} - ${bookings_df['booked_price'].max():.2f}"
        )


# Example usage
if __name__ == "__main__":
    # Load sample search data
    print("Loading sample search data...")
    searches_df = pd.read_csv("data/realistic_searches.csv")

    # Ensure datetime
    searches_df["search_ts"] = pd.to_datetime(searches_df["search_ts"])

    # Add location_id if missing (for compatibility)
    if "location_id" not in searches_df.columns:
        searches_df["location_id"] = np.random.randint(1, 25, len(searches_df))

    # Generate bookings
    generator = RealisticBookingGenerator()
    bookings_df = generator.generate_bookings(searches_df)

    # Save bookings
    output_path = "data/realistic_bookings.csv"
    bookings_df.to_csv(output_path, index=False)
    print(f"\nBookings saved to: {output_path}")

"""
Realistic car rental and competitor price generator with temporal patterns.

This module generates synthetic pricing data based on real-world patterns:
- Seasonal variations (summer peaks, winter lows)
- Day of week patterns (weekday business vs weekend leisure)
- Supplier pricing tiers
- Location premiums (airport vs downtown)
- Dynamic pricing based on supply/demand
- Event-based pricing spikes
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


class RealisticPriceGenerator:
    """
    Generates realistic car rental pricing data with temporal and market patterns.

    Key features:
    - Seasonal pricing variations
    - Day of week patterns
    - Supplier-specific pricing tiers
    - Location-based premiums
    - Dynamic supply/demand adjustments
    - Event and holiday pricing spikes
    """

    def __init__(self, seed: int = 2025):
        """
        Initialize the price generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

        # Base prices by car class (economy baseline)
        self.base_prices = {
            "economy": 45.0,
            "compact": 52.0,
            "suv": 75.0,
            "luxury": 120.0,
        }

        # Supplier pricing tiers (multipliers)
        # Based on research: Budget cheapest, National most expensive
        self.supplier_multipliers = {
            1: 1.15,  # Enterprise (mid-tier)
            2: 1.20,  # Hertz (premium)
            3: 1.18,  # Avis (mid-premium)
            4: 0.95,  # Budget (economy)
            5: 1.10,  # Sixt (value premium)
        }

        # Supplier names for reference
        self.supplier_names = {
            1: "Enterprise",
            2: "Hertz",
            3: "Avis",
            4: "Budget",
            5: "Sixt",
        }

        # Monthly multipliers based on research
        self.monthly_multipliers = {
            1: 0.80,  # January - lowest
            2: 0.85,  # February - low
            3: 0.95,  # March - spring break begins
            4: 1.00,  # April - moderate
            5: 1.15,  # May - pre-summer high
            6: 1.25,  # June - summer peak begins
            7: 1.30,  # July - peak summer
            8: 1.25,  # August - still peak
            9: 1.05,  # September - post-summer
            10: 0.95,  # October - moderate
            11: 0.90,  # November - low season
            12: 1.10,  # December - holiday travel
        }

        # Day of week multipliers
        self.dow_multipliers = {
            0: 1.10,  # Monday - business travel
            1: 1.15,  # Tuesday - peak business
            2: 1.15,  # Wednesday - peak business
            3: 1.12,  # Thursday - business + early weekend
            4: 1.05,  # Friday - mixed
            5: 0.95,  # Saturday - leisure discount
            6: 0.95,  # Sunday - leisure discount
        }

        # Location premiums (airport locations)
        self.airport_locations = {1, 5, 9, 13, 17, 21}  # Major airports
        self.airport_premium = 1.22  # 22% premium for airports

        # Major holidays and events
        self.holidays = self._define_holidays()

    def _define_holidays(self) -> dict[tuple[int, int], float]:
        """Define major holidays and their price multipliers."""
        return {
            (1, 1): 1.5,  # New Year's Day
            (2, 14): 1.3,  # Valentine's Day
            (3, 17): 1.4,  # St. Patrick's Day (varies)
            (5, 25): 1.4,  # Memorial Day weekend (last Monday)
            (7, 4): 1.5,  # July 4th
            (9, 1): 1.4,  # Labor Day (first Monday)
            (11, 22): 1.6,  # Thanksgiving (4th Thursday)
            (12, 25): 1.8,  # Christmas
            (12, 31): 1.7,  # New Year's Eve
        }

    def generate_rental_prices(
        self,
        n_records: int,
        start_date: str,
        end_date: str,
        locations: list[int],
        suppliers: list[int],
        car_classes: list[str],
    ) -> pd.DataFrame:
        """
        Generate hourly rental price observations.

        Args:
            n_records: Number of price records to generate
            start_date: Start date for observations
            end_date: End date for observations
            locations: List of valid location IDs
            suppliers: List of valid supplier IDs
            car_classes: List of valid car classes

        Returns:
            DataFrame with rental price observations
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate observation timestamps (hourly)
        date_range = pd.date_range(start_dt, end_dt, freq="h")
        obs_timestamps = self.rng.choice(date_range, size=n_records)

        # Generate pickup dates (future dates from observation)
        # Most observations are for pickups 1-60 days in the future
        days_ahead = self.rng.choice(
            np.arange(1, 61),
            size=n_records,
            p=self._get_booking_lead_time_distribution(),
        )

        prices = []
        for i in range(n_records):
            obs_ts = pd.Timestamp(obs_timestamps[i])
            pickup_date = obs_ts + timedelta(days=int(days_ahead[i]))

            # Select attributes
            location_id = self.rng.choice(locations)
            supplier_id = self.rng.choice(suppliers)
            car_class = self.rng.choice(car_classes)

            # Calculate price
            price = self._calculate_price(
                pickup_date, location_id, supplier_id, car_class, obs_ts, days_ahead[i]
            )

            # Calculate availability (inverse relationship with price)
            # Higher prices = lower availability
            base_availability = 15
            price_factor = price / self.base_prices[car_class]
            availability = max(
                0, int(base_availability / price_factor + self.rng.normal(0, 3))
            )

            prices.append(
                {
                    "price_id": i + 1,
                    "location_id": location_id,
                    "supplier_id": supplier_id,
                    "car_class": car_class,
                    "pickup_date": pickup_date,
                    "obs_ts": obs_ts,
                    "current_price": round(price, 2),
                    "available_cars": availability,
                    "days_until_pickup": int(days_ahead[i]),
                }
            )

        df = pd.DataFrame(prices)

        # Sort by observation timestamp
        df = df.sort_values("obs_ts").reset_index(drop=True)
        df["price_id"] = range(1, len(df) + 1)

        return df

    def generate_competitor_prices(
        self,
        n_records: int,
        start_date: str,
        end_date: str,
        locations: list[int],
        car_classes: list[str],
    ) -> pd.DataFrame:
        """
        Generate daily competitor price observations (minimum across all suppliers).

        Args:
            n_records: Number of price records to generate
            start_date: Start date for observations
            end_date: End date for observations
            locations: List of valid location IDs
            car_classes: List of valid car classes

        Returns:
            DataFrame with competitor price observations
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate observation dates (daily)
        date_range = pd.date_range(start_dt, end_dt, freq="D")
        obs_dates = self.rng.choice(date_range, size=n_records)

        # Generate pickup dates
        days_ahead = self.rng.choice(
            np.arange(1, 61),
            size=n_records,
            p=self._get_booking_lead_time_distribution(),
        )

        comp_prices = []
        for i in range(n_records):
            obs_date = pd.Timestamp(obs_dates[i])
            pickup_date = obs_date + timedelta(days=int(days_ahead[i]))

            # Select attributes
            location_id = self.rng.choice(locations)
            car_class = self.rng.choice(car_classes)

            # Calculate minimum price across all suppliers
            supplier_prices = []
            for supplier_id in range(1, 6):  # All 5 suppliers
                price = self._calculate_price(
                    pickup_date,
                    location_id,
                    supplier_id,
                    car_class,
                    obs_date,
                    days_ahead[i],
                )
                supplier_prices.append(price)

            # Competitor price is minimum with some noise
            min_price = min(supplier_prices) * self.rng.uniform(0.95, 1.02)

            comp_prices.append(
                {
                    "comp_id": i + 1,
                    "location_id": location_id,
                    "car_class": car_class,
                    "pickup_date": pickup_date,
                    "obs_date": obs_date,
                    "comp_min_price": round(min_price, 2),
                    "days_until_pickup": int(days_ahead[i]),
                }
            )

        df = pd.DataFrame(comp_prices)

        # Sort by observation date
        df = df.sort_values("obs_date").reset_index(drop=True)
        df["comp_id"] = range(1, len(df) + 1)

        return df

    def _calculate_price(
        self,
        pickup_date: pd.Timestamp,
        location_id: int,
        supplier_id: int,
        car_class: str,
        obs_time: pd.Timestamp,
        days_until_pickup: float,
    ) -> float:
        """Calculate price based on all factors."""
        # Start with base price
        base = self.base_prices[car_class]

        # Apply supplier multiplier
        base *= self.supplier_multipliers[supplier_id]

        # Apply monthly seasonality
        base *= self.monthly_multipliers[pickup_date.month]

        # Apply day of week pattern
        base *= self.dow_multipliers[pickup_date.weekday()]

        # Apply location premium
        if location_id in self.airport_locations:
            base *= self.airport_premium

        # Apply holiday premium
        holiday_mult = self._get_holiday_multiplier(pickup_date)
        base *= holiday_mult

        # Apply booking lead time adjustment
        # Last-minute bookings (< 7 days) are more expensive
        if days_until_pickup < 7:
            base *= 1.15 + (7 - days_until_pickup) * 0.03
        elif days_until_pickup > 45:
            # Far advance bookings get small discount
            base *= 0.95

        # Apply supply/demand variation (random ±15%)
        base *= self.rng.uniform(0.85, 1.15)

        # Apply time of day variation for obs_time
        hour = obs_time.hour
        if 6 <= hour <= 9 or 17 <= hour <= 20:  # Peak hours
            base *= 1.05
        elif 0 <= hour <= 5:  # Night hours
            base *= 0.98

        return max(25.0, base)  # Minimum price floor

    def _get_booking_lead_time_distribution(self) -> np.ndarray:
        """Get probability distribution for booking lead times."""
        # Based on research: most bookings happen 1-30 days out
        # Create a distribution that favors 7-21 days
        days = np.arange(1, 61)

        # Peak around 14 days, decay after
        weights = np.exp(-0.5 * ((days - 14) / 10) ** 2)

        # Boost last-minute bookings slightly
        weights[:7] *= 1.5

        # Normalize
        return weights / weights.sum()

    def _get_holiday_multiplier(self, date: pd.Timestamp) -> float:
        """Get holiday price multiplier for a given date."""
        # Check exact date
        date_key = (date.month, date.day)
        if date_key in self.holidays:
            return self.holidays[date_key]

        # Check if near a holiday (within 3 days)
        for holiday_date, multiplier in self.holidays.items():
            holiday = pd.Timestamp(date.year, holiday_date[0], holiday_date[1])
            if abs((date - holiday).days) <= 3:
                # Gradual increase/decrease around holidays
                distance = abs((date - holiday).days)
                return 1.0 + (multiplier - 1.0) * (1 - distance / 4)

        return 1.0

    def generate_price_summary(self, prices_df: pd.DataFrame) -> None:
        """Print summary statistics for generated prices."""
        print("\n" + "=" * 60)
        print("PRICE GENERATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal price records: {len(prices_df):,}")

        # Price statistics by car class
        print("\nAverage prices by car class:")
        for car_class in ["economy", "compact", "suv", "luxury"]:
            if car_class in prices_df["car_class"].values:
                avg_price = prices_df[prices_df["car_class"] == car_class][
                    "current_price"
                ].mean()
                print(f"  {car_class}: ${avg_price:.2f}")

        # Price statistics by supplier
        print("\nAverage prices by supplier:")
        for supplier_id in sorted(prices_df["supplier_id"].unique()):
            avg_price = prices_df[prices_df["supplier_id"] == supplier_id][
                "current_price"
            ].mean()
            supplier_name = self.supplier_names.get(
                supplier_id, f"Supplier {supplier_id}"
            )
            print(f"  {supplier_name}: ${avg_price:.2f}")

        # Monthly patterns
        if "pickup_date" in prices_df.columns:
            prices_df["pickup_month"] = pd.to_datetime(
                prices_df["pickup_date"]
            ).dt.month
            print("\nAverage prices by month:")
            monthly_avg = (
                prices_df.groupby("pickup_month")["current_price"].mean().sort_index()
            )
            for month, price in monthly_avg.items():
                month_name = pd.Timestamp(2024, int(month), 1).strftime("%B")
                print(f"  {month_name}: ${price:.2f}")

        # Location patterns
        print("\nAirport vs Downtown prices:")
        airport_prices = prices_df[
            prices_df["location_id"].isin(self.airport_locations)
        ]["current_price"].mean()
        downtown_prices = prices_df[
            ~prices_df["location_id"].isin(self.airport_locations)
        ]["current_price"].mean()
        print(f"  Airport locations: ${airport_prices:.2f}")
        print(f"  Downtown locations: ${downtown_prices:.2f}")
        print(f"  Airport premium: {(airport_prices / downtown_prices - 1) * 100:.1f}%")


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = RealisticPriceGenerator()

    # Generate rental prices
    print("Generating rental prices...")
    rental_prices = generator.generate_rental_prices(
        n_records=30000,
        start_date="2024-01-01",
        end_date="2024-12-31",
        locations=list(range(1, 25)),
        suppliers=[1, 2, 3, 4, 5],
        car_classes=["economy", "compact", "suv", "luxury"],
    )

    # Generate competitor prices
    print("\nGenerating competitor prices...")
    competitor_prices = generator.generate_competitor_prices(
        n_records=10000,
        start_date="2024-01-01",
        end_date="2024-12-31",
        locations=list(range(1, 25)),
        car_classes=["economy", "compact", "suv", "luxury"],
    )

    # Print summaries
    generator.generate_price_summary(rental_prices)

    # Save data
    rental_prices.to_csv("data/realistic_rental_prices.csv", index=False)
    competitor_prices.to_csv("data/realistic_competitor_prices.csv", index=False)

    print(f"\n✅ Generated {len(rental_prices):,} rental prices")
    print(f"✅ Generated {len(competitor_prices):,} competitor prices")
    print("\nFiles saved to:")
    print("  - data/realistic_rental_prices.csv")
    print("  - data/realistic_competitor_prices.csv")

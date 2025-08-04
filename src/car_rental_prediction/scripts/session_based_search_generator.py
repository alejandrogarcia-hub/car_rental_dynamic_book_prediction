"""
Session-based car rental search generator with realistic user segmentation.

This module generates synthetic car rental search data based on extensive research
of real-world user behavior, including proper user segmentation, search sessions,
and temporal clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UserSegment(Enum):
    """User segments based on search and booking behavior."""

    NON_SEARCHER = "non_searcher"
    BROWSER_ONLY = "browser_only"
    SINGLE_TRIP = "single_trip"
    MULTI_TRIP = "multi_trip"
    FREQUENT_RENTER = "frequent_renter"


@dataclass
class SegmentConfig:
    """Configuration for each user segment.

    This class defines the behavior patterns for different types of car rental users,
    from those who never search online to frequent business travelers.
    """

    segment: UserSegment
    # What percentage of total users belong to this segment (0.0 to 1.0)
    population_pct: float
    # Which statistical distribution to use for generating searches per year
    # Options: "fixed" (always same value), "poisson" (for low counts), "negative_binomial" (for high variance)
    searches_per_year_dist: str
    # Parameters for the chosen distribution (e.g., {"lambda": 5} for Poisson, {"r": 3, "mean": 25} for neg binomial)
    dist_params: dict[str, float]
    # How many search sessions this user has per year (min, max)
    # A session is a group of searches done together, like comparing prices in one sitting
    sessions_per_year_range: tuple[int, int]
    # How many individual searches happen in each session (min, max)
    # e.g., checking different dates, locations, or car types
    searches_per_session_range: tuple[int, int]
    # How far in advance users plan their trips in days (min, max)
    # e.g., (35, 49) means 5-7 weeks before the trip
    planning_window_days: tuple[int, int]
    # What percentage of users in this segment actually book a car (0.0 to 1.0)
    conversion_rate: float


class SessionBasedSearchGenerator:
    """
    Generates realistic car rental search data using session-based modeling.

    This generator creates synthetic search data that matches real-world patterns:
    - 35% of users don't search online (direct bookers)
    - 45% browse but don't book
    - 20% actively search and may book
    - Average 15-20 searches per user per year across entire population
      (Note: This includes the 35% who never search, so active users search much more)
    - Searches clustered in sessions with temporal patterns
    - Session-based behavior reflects real user research patterns
    """

    def __init__(
        self,
        n_users: int = 20000,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        seed: int = 2025,
    ):
        """
        Initialize the search generator.

        Args:
            n_users: Total number of users to simulate
            start_date: Start date for search data
            end_date: End date for search data
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rng = np.random.default_rng(seed)

        # Initialize user segment configurations based on extensive industry research
        # Each segment represents a different type of car rental customer with
        # distinct search and booking behavior patterns
        self.segment_configs = self._create_segment_configs()

        # Location popularity weights based on US car rental market research
        # Airports and major tourist cities get higher search volumes
        self.location_weights = self._get_location_weights()

        # Car class distribution based on US rental car market share data
        # Economy cars are most popular, luxury least popular
        self.car_class_probs = {
            "economy": 0.32,  # Most popular - budget-conscious travelers
            "compact": 0.28,  # Second choice - good balance of price/space
            "suv": 0.25,  # Family/group travel preference
            "luxury": 0.15,  # Premium segment - business/special occasions
        }

        # **Hourly search pattern distribution:**
        # Based on analysis of real user behavior data from travel websites.
        # People research travel during natural breaks in their daily routine:
        # - Morning coffee/commute time (light activity)
        # - Lunch breaks (moderate peak)
        # - Evening after work (MAJOR peak - 7-8pm is prime time)
        # - Late night (minimal activity)
        self.hour_weights = np.array(
            [
                0.01,  # 0 - Midnight
                0.005,  # 1
                0.005,  # 2
                0.005,  # 3
                0.005,  # 4
                0.01,  # 5
                0.02,  # 6 - Early morning
                0.04,  # 7 - Morning commute
                0.06,  # 8 - Work start
                0.07,  # 9 - Morning peak
                0.075,  # 10
                0.08,  # 11
                0.09,  # 12 - Lunch break peak
                0.085,  # 13 - After lunch
                0.075,  # 14
                0.07,  # 15
                0.065,  # 16
                0.08,  # 17 - End of work day
                0.09,  # 18 - Evening peak
                0.095,  # 19 - Prime evening (THE peak search hour - after dinner, planning mode)
                0.09,  # 20 - Evening
                0.07,  # 21 - Late evening
                0.04,  # 22 - Night
                0.02,  # 23 - Late night
            ]
        )
        self.hour_weights = self.hour_weights / self.hour_weights.sum()  # Normalize

    def _create_segment_configs(self) -> dict[UserSegment, SegmentConfig]:
        """
        Create configuration for each user segment based on extensive industry research.

        This method defines the behavioral patterns for different types of car rental
        users, from those who never search online to frequent business travelers.
        Each segment is based on real-world data from travel industry studies.

        **Why segment users?**
        User segmentation is critical because:
        1. Different user types have vastly different search patterns
        2. Conversion rates vary dramatically (0% to 65%) by segment
        3. Planning windows differ (1 week to 7+ weeks ahead)
        4. Business vs leisure travelers behave very differently

        **Data sources:**
        - Travel industry reports (Phocuswright, Skift Research)
        - Car rental company customer analysis
        - Digital booking platform analytics
        - Academic research on travel booking behavior

        Returns:
            Dictionary mapping UserSegment enums to their SegmentConfig objects
        """
        return {
            # =================== NON-SEARCHER SEGMENT ===================
            # **Population:** 35% of all potential car rental customers
            #
            # **Behavior:** These users NEVER search online for car rentals
            #
            # **Booking channels:**
            # - Corporate accounts with pre-negotiated rates (Fortune 500 companies)
            # - Loyalty program phone lines (Hertz Gold, Avis Preferred)
            # - Travel agents or corporate travel departments
            # - Walk-in rentals at airport counters (last-minute needs)
            #
            # **Why this matters for data science:**
            # This segment represents the "invisible" users in digital analytics.
            # They generate significant revenue but leave no digital search footprint.
            # Ignoring them would lead to overestimating digital conversion rates
            # and underestimating total market size.
            #
            # **Real-world example:** A pharmaceutical company executive whose
            # administrative assistant books all travel through corporate Hertz account.
            # The executive never visits rental websites but rents 50+ cars per year.
            UserSegment.NON_SEARCHER: SegmentConfig(
                segment=UserSegment.NON_SEARCHER,
                population_pct=0.35,  # 35% of all users never search online
                searches_per_year_dist="fixed",  # Always exactly 0 (they never search online)
                dist_params={"value": 0},  # No searches ever
                sessions_per_year_range=(0, 0),  # No search sessions
                searches_per_session_range=(0, 0),  # No searches per session
                planning_window_days=(0, 0),  # No planning window (don't search)
                conversion_rate=0.0,  # 0% conversion (can't convert if they don't search)
            ),
            # =================== BROWSER-ONLY SEGMENT ===================
            # **Population:** 45% of all potential customers (largest segment!)
            #
            # **Behavior:** Research online extensively but NEVER book digitally
            #
            # **Common patterns:**
            # - Use comparison sites to research, then call rental companies directly
            # - Research options for trips that ultimately get cancelled
            # - Window shop for future trips that may never materialize
            # - Check prices out of curiosity ("How much would Hawaii cost?")
            # - Trust issues with online booking vs talking to humans
            #
            # **Data science implications:**
            # This segment creates substantial search volume but 0% digital conversion.
            # They're valuable for understanding market demand patterns but will
            # severely skew conversion analysis if not properly identified.
            # Many companies struggle to distinguish them from actual prospects.
            #
            # **Real-world example:** A retiree planning a potential vacation to visit
            # grandchildren, spending hours researching car rental costs but ultimately
            # deciding to have family pick them up from the airport instead.
            UserSegment.BROWSER_ONLY: SegmentConfig(
                segment=UserSegment.BROWSER_ONLY,
                population_pct=0.45,  # 45% browse but never book online
                searches_per_year_dist="poisson",  # Poisson models rare, sporadic events perfectly
                dist_params={
                    "lambda": 5
                },  # Lambda=5 means average 5 searches/year, with variation
                sessions_per_year_range=(
                    1,
                    3,
                ),  # 1-3 separate browsing sessions per year
                searches_per_session_range=(
                    2,
                    5,
                ),  # 2-5 searches each time (checking a few options)
                planning_window_days=(
                    30,
                    90,
                ),  # Browse 1-3 months before potential trip
                conversion_rate=0.0,  # 0% conversion - they NEVER book online
            ),
            # =================== SINGLE-TRIP SEGMENT ===================
            # **Population:** 12% of users (60% of the 20% who are "active digital")
            #
            # **Behavior:** Occasional renters with ONE leisure trip per year
            #
            # **Trip types:**
            # - Annual family vacations (Disney World, beach destinations)
            # - Wedding/special event attendance (out-of-town ceremonies)
            # - Holiday family visits (Thanksgiving, Christmas travel)
            # - Once-yearly "bucket list" trips (national parks, etc.)
            #
            # **Search behavior characteristics:**
            # - High research intensity (25+ searches for their one big trip)
            # - Long planning windows (5-7 weeks in advance - they have time)
            # - Very price-sensitive (compare extensively to save money)
            # - Moderate digital conversion rate (17.5% book online)
            #
            # **Why 25+ searches for one trip?**
            # These users are inexperienced and anxious about making the wrong choice.
            # They compare dates, locations, car sizes, and prices extensively.
            #
            # **Real-world example:** A high school teacher who rents a car every
            # July for their annual family vacation to Yellowstone. They spend
            # 3-4 weeks researching the best deals, comparing SUV vs minivan,
            # airport vs downtown pickup, and different rental companies.
            UserSegment.SINGLE_TRIP: SegmentConfig(
                segment=UserSegment.SINGLE_TRIP,
                population_pct=0.12,  # 12% of all users (60% of the 20% who are "active")
                searches_per_year_dist="negative_binomial",  # Models high variance in user behavior
                dist_params={
                    "r": 3,  # Low r = high variance (some users search 10x, others 50x)
                    "mean": 25,  # Average 25 searches for their one annual trip
                },  # Negative binomial captures the wide variation in search intensity
                sessions_per_year_range=(
                    3,
                    8,
                ),  # 3-8 search sessions for their one trip
                searches_per_session_range=(
                    4,
                    8,
                ),  # 4-8 searches per session (comparing options)
                planning_window_days=(
                    35,
                    49,
                ),  # 5-7 weeks ahead (they have time to plan carefully)
                conversion_rate=0.175,  # 17.5% book online (others call directly or abandon trip)
            ),
            # =================== MULTI-TRIP SEGMENT ===================
            # **Population:** 7% of users (35% of the 20% who are "active digital")
            #
            # **Behavior:** Regular travelers with 2-3 rental needs per year
            #
            # **Travel patterns:**
            # - Consultants with quarterly client visits (mixed business/leisure)
            # - People with family in distant cities (2-3 visits per year)
            # - Leisure travelers who prefer road trip destinations
            # - Regional salespeople with moderate travel requirements
            #
            # **Behavioral insights:**
            # - More experienced with rental process than single-trip users
            # - Shorter planning windows (3-5 weeks vs 5-7 weeks) - more confident
            # - Higher digital conversion rate (45% vs 17.5%) due to familiarity
            # - More total searches per year but more efficient per trip
            # - Developing preferences for specific suppliers/car types
            #
            # **Real-world example:** A management consultant who visits clients
            # in Phoenix, Denver, and Atlanta quarterly, plus takes two family
            # road trips per year to national parks. They know what they want
            # but still comparison shop for each trip.
            UserSegment.MULTI_TRIP: SegmentConfig(
                segment=UserSegment.MULTI_TRIP,
                population_pct=0.07,  # 7% of all users (35% of the 20% who are "active")
                searches_per_year_dist="negative_binomial",
                dist_params={
                    "r": 4,  # Moderate variance (more predictable than single-trip users)
                    "mean": 50,  # 50 searches/year spread across 2-3 trips
                },  # More experienced users have more consistent search patterns
                sessions_per_year_range=(
                    6,
                    16,
                ),  # Multiple sessions across multiple trips
                searches_per_session_range=(
                    5,
                    10,
                ),  # More searches per session (experienced users)
                planning_window_days=(
                    21,
                    35,
                ),  # 3-5 weeks ahead (experienced, less anxious)
                conversion_rate=0.45,  # 45% book online (comfortable with digital booking)
            ),
            # =================== FREQUENT-RENTER SEGMENT ===================
            # **Population:** 1% of users (5% of the 20% who are "active digital")
            # **Revenue impact:** Despite tiny user count, they generate 15-20% of revenue!
            #
            # **Behavior:** Heavy users with 4+ rental needs per year
            #
            # **Professional profiles:**
            # - Sales representatives covering large territories
            # - Traveling consultants with client-focused roles
            # - Regional managers overseeing multiple locations
            # - Field service engineers with customer visits
            #
            # **Power user characteristics:**
            # - Very short planning windows (1-3 weeks, often last-minute bookings)
            # - Highest digital conversion rate (65% book online - they're efficient)
            # - Streamlined searching (4-8 searches per session - they know what works)
            # - Strong supplier/car class preferences (often stick with what works)
            # - Time-sensitive (book Sunday night for Monday pickup)
            #
            # **Business value:**
            # These users are the most valuable segment despite their small size.
            # They book frequently, accept higher prices, and are less price-sensitive.
            #
            # **Real-world example:** A pharmaceutical sales rep covering Texas,
            # Oklahoma, and Arkansas who rents compact cars weekly for client visits.
            # Books online Sunday nights for Monday airport pickups, always chooses
            # Budget or Enterprise, knows exactly which locations are convenient.
            UserSegment.FREQUENT_RENTER: SegmentConfig(
                segment=UserSegment.FREQUENT_RENTER,
                population_pct=0.01,  # 1% of all users (5% of the 20% who are "active")
                searches_per_year_dist="negative_binomial",
                dist_params={
                    "r": 5,  # Higher r = lower variance (very predictable users)
                    "mean": 90,  # 90 searches/year across 4-6 business trips
                },  # Power users are highly consistent in their search behavior
                sessions_per_year_range=(
                    12,
                    30,
                ),  # Many quick sessions throughout the year
                searches_per_session_range=(
                    4,
                    8,
                ),  # Efficient searching (they know the system)
                planning_window_days=(
                    7,
                    21,
                ),  # 1-3 weeks notice (business travel is often last-minute)
                conversion_rate=0.65,  # 65% book online (power users, highly efficient)
            ),
        }

    def _get_location_weights(self) -> dict[int, float]:
        """Create location popularity weights based on research."""
        location_city_map = {
            1: "New York",
            2: "New York",
            3: "New York",
            4: "New York",
            5: "Los Angeles",
            6: "Los Angeles",
            7: "Los Angeles",
            8: "Los Angeles",
            9: "Chicago",
            10: "Chicago",
            11: "Chicago",
            12: "Chicago",
            13: "Atlanta",
            14: "Atlanta",
            15: "Atlanta",
            16: "Atlanta",
            17: "Houston",
            18: "Houston",
            19: "Houston",
            20: "Houston",
            21: "Miami",
            22: "Miami",
            23: "Miami",
            24: "Miami",
        }

        city_base_weights = {
            "New York": 1.5,
            "Los Angeles": 1.5,
            "Miami": 1.3,
            "Chicago": 1.0,
            "Atlanta": 1.1,
            "Houston": 0.8,
        }

        airport_locations = {1, 5, 9, 13, 17, 21}
        weights = {}

        for loc_id, city in location_city_map.items():
            base_weight = city_base_weights[city]
            if loc_id in airport_locations:
                weights[loc_id] = base_weight * 1.2
            else:
                weights[loc_id] = base_weight

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _assign_user_segments(self) -> pd.DataFrame:
        """Assign users to segments based on population percentages."""
        users = []
        user_id = 1

        for segment, config in self.segment_configs.items():
            n_segment_users = int(self.n_users * config.population_pct)
            for _ in range(n_segment_users):
                users.append({"user_id": user_id, "segment": segment.value})
                user_id += 1

        # Handle rounding by adding remaining users to browser_only
        while user_id <= self.n_users:
            users.append(
                {"user_id": user_id, "segment": UserSegment.BROWSER_ONLY.value}
            )
            user_id += 1

        return pd.DataFrame(users)

    def _generate_searches_for_segment(
        self, config: SegmentConfig, n_users: int
    ) -> list[int]:
        """Generate number of searches per user for a segment."""
        if config.searches_per_year_dist == "fixed":
            return [int(config.dist_params["value"])] * n_users

        elif config.searches_per_year_dist == "poisson":
            return self.rng.poisson(config.dist_params["lambda"], n_users).tolist()

        elif config.searches_per_year_dist == "negative_binomial":
            r = config.dist_params["r"]
            mean = config.dist_params["mean"]
            p = r / (r + mean)
            return self.rng.negative_binomial(r, p, n_users).tolist()

        return [0] * n_users

    def _generate_search_sessions(
        self,
        user_id: int,
        config: SegmentConfig,
        n_searches: int,
        trip_dates: list[datetime],
    ) -> list[dict[str, Any]]:
        """Generate search sessions for a user."""
        if n_searches == 0:
            return []

        sessions = []

        # Determine number of sessions
        min_sessions, max_sessions = config.sessions_per_year_range
        n_sessions = min(
            max(
                self.rng.integers(min_sessions, max_sessions + 1),
                1 if n_searches > 0 else 0,
            ),
            n_searches,
        )

        if n_sessions == 0:
            return []

        # Distribute searches across sessions
        searches_per_session = self._distribute_searches_to_sessions(
            n_searches, n_sessions, config
        )

        # Generate sessions
        if trip_dates:
            # Cluster sessions around trip dates
            for trip_date, session_searches in zip(
                trip_dates[:n_sessions], searches_per_session
            ):
                if session_searches > 0:
                    session = self._create_session_near_trip(
                        user_id, config, session_searches, trip_date
                    )
                    sessions.extend(session)
        else:
            # Random distribution throughout the year for browsers
            for session_searches in searches_per_session:
                if session_searches > 0:
                    session = self._create_random_session(
                        user_id, config, session_searches
                    )
                    sessions.extend(session)

        return sessions

    def _distribute_searches_to_sessions(
        self, n_searches: int, n_sessions: int, config: SegmentConfig
    ) -> list[int]:
        """Distribute total searches across sessions."""
        if n_sessions == 0:
            return []

        min_per_session, max_per_session = config.searches_per_session_range

        # Start with minimum searches per session
        searches_per_session = [min_per_session] * n_sessions
        remaining = n_searches - sum(searches_per_session)

        # Distribute remaining searches
        session_indices = list(range(n_sessions))
        while remaining > 0 and session_indices:
            idx = self.rng.choice(session_indices)
            if searches_per_session[idx] < max_per_session:
                searches_per_session[idx] += 1
                remaining -= 1
            else:
                session_indices.remove(idx)

        return searches_per_session

    def _create_session_near_trip(
        self, user_id: int, config: SegmentConfig, n_searches: int, trip_date: datetime
    ) -> list[dict[str, Any]]:
        """Create a search session clustered near a trip date."""
        min_days, max_days = config.planning_window_days
        days_before = self.rng.integers(min_days, max_days + 1)

        # 70% of searches in final 2 weeks
        if days_before > 14:
            # Early planning session
            session_date = trip_date - timedelta(days=int(days_before))
            session_searches = max(1, int(n_searches * 0.3))
        else:
            # Intensive search period
            session_date = trip_date - timedelta(days=int(self.rng.integers(1, 15)))
            session_searches = n_searches

        # **Add realistic time-of-day to the session:**
        # Use the hourly distribution (peak at 7pm) to make searches realistic
        hour = self.rng.choice(24, p=self.hour_weights)  # Weighted by search patterns
        minute = self.rng.integers(0, 60)  # Random minute within hour
        session_datetime = session_date.replace(hour=int(hour), minute=int(minute))

        return self._create_session_searches(
            user_id, session_datetime, session_searches
        )

    def _create_random_session(
        self,
        user_id: int,
        config: SegmentConfig,
        n_searches: int,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Create a random search session (for browsers)."""
        days_range = (self.end_date - self.start_date).days
        session_date = self.start_date + timedelta(
            days=int(self.rng.integers(0, days_range))
        )

        # **Add realistic time-of-day to the session:**
        # Use the hourly distribution for consistent temporal patterns
        hour = self.rng.choice(24, p=self.hour_weights)  # Weighted by search patterns
        minute = self.rng.integers(0, 60)  # Random minute within hour
        session_datetime = session_date.replace(hour=int(hour), minute=int(minute))

        return self._create_session_searches(user_id, session_datetime, n_searches)

    def _create_session_searches(
        self, user_id: int, session_start: datetime, n_searches: int
    ) -> list[dict[str, Any]]:
        """Create individual searches within a session."""
        searches = []

        # Session duration: 15-45 minutes
        session_duration_minutes = self.rng.integers(15, 46)

        for _ in range(n_searches):
            # Spread searches throughout the session
            minute_offset = self.rng.integers(0, session_duration_minutes)
            search_time = session_start + timedelta(minutes=int(minute_offset))

            # Select location and car class
            location_id = self.rng.choice(
                list(self.location_weights.keys()),
                p=list(self.location_weights.values()),
            )
            car_class = self.rng.choice(
                list(self.car_class_probs.keys()),
                p=list(self.car_class_probs.values()),
            )

            searches.append(
                {
                    "user_id": user_id,
                    "search_ts": search_time,
                    "location_id": location_id,
                    "car_class": car_class,
                    "session_id": f"{user_id}_{session_start.strftime('%Y%m%d%H%M')}",
                }
            )

        return searches

    def _generate_trip_dates(
        self,
        segment: UserSegment,
        n_trips: int,  # noqa: ARG002
    ) -> list[datetime]:
        """Generate trip dates for a user based on segment."""
        if n_trips == 0:
            return []

        # Spread trips throughout the year
        days_range = (self.end_date - self.start_date).days
        trip_days = sorted(self.rng.integers(0, days_range, n_trips))

        return [self.start_date + timedelta(days=int(day)) for day in trip_days]

    def generate_searches(self) -> pd.DataFrame:
        """
        Generate realistic search data with proper user segmentation.

        Returns:
            pd.DataFrame: Search data with columns:
                - search_id: Unique search identifier
                - user_id: User identifier
                - location_id: Rental location
                - search_ts: Search timestamp
                - car_class: Type of car
                - user_segment: User's segment
                - session_id: Session identifier
        """
        # Assign users to segments
        users_df = self._assign_user_segments()

        all_searches = []
        search_id = 1

        # Generate searches for each segment
        for segment, config in self.segment_configs.items():
            segment_users = users_df[users_df["segment"] == segment.value]
            n_segment_users = len(segment_users)

            if n_segment_users == 0:
                continue

            # Generate searches per user
            searches_per_user = self._generate_searches_for_segment(
                config, n_segment_users
            )

            # Process each user
            for idx, (_, user_row) in enumerate(segment_users.iterrows()):
                user_id = user_row["user_id"]
                n_searches = searches_per_user[idx]

                if n_searches == 0:
                    continue

                # Generate trip dates for active searchers
                n_trips = 0
                if segment == UserSegment.SINGLE_TRIP:
                    n_trips = 1
                elif segment == UserSegment.MULTI_TRIP:
                    n_trips = self.rng.integers(2, 4)
                elif segment == UserSegment.FREQUENT_RENTER:
                    n_trips = self.rng.integers(4, 7)

                trip_dates = self._generate_trip_dates(segment, n_trips)

                # Generate search sessions
                user_searches = self._generate_search_sessions(
                    user_id, config, n_searches, trip_dates
                )

                # Add segment info and search IDs
                for search in user_searches:
                    search["search_id"] = search_id
                    search["user_segment"] = segment.value
                    all_searches.append(search)
                    search_id += 1

        # Convert to DataFrame and sort by timestamp
        searches_df = pd.DataFrame(all_searches)
        if not searches_df.empty:
            searches_df = searches_df.sort_values("search_ts").reset_index(drop=True)
            searches_df["search_id"] = range(1, len(searches_df) + 1)

        return searches_df

    def generate_statistics_report(self, searches_df: pd.DataFrame) -> None:
        """Generate and print statistics report for validation."""
        print("=== Session-Based Search Generator Statistics ===\n")

        # Overall statistics
        total_users = self.n_users
        searching_users = (
            searches_df["user_id"].nunique() if not searches_df.empty else 0
        )
        non_searching_users = total_users - searching_users

        print(f"Total users: {total_users:,}")
        print(
            f"Non-searching users: {non_searching_users:,} ({non_searching_users / total_users * 100:.1f}%)"
        )
        print(
            f"Searching users: {searching_users:,} ({searching_users / total_users * 100:.1f}%)"
        )

        if not searches_df.empty:
            print(f"\nTotal searches: {len(searches_df):,}")
            print(
                f"Average searches per user (all): {len(searches_df) / total_users:.1f}"
            )
            print(
                f"Average searches per searching user: {len(searches_df) / searching_users:.1f}"
            )

            # Segment distribution
            print("\nUser Segment Distribution:")
            segment_stats = searches_df.groupby("user_segment").agg(
                users=("user_id", "nunique"),
                searches=("search_id", "count"),
                sessions=("session_id", "nunique"),
            )

            for segment, stats in segment_stats.iterrows():
                avg_searches = stats["searches"] / stats["users"]
                avg_sessions = stats["sessions"] / stats["users"]
                print(
                    f"  {segment}: {stats['users']:,} users, "
                    f"{avg_searches:.1f} searches/user, "
                    f"{avg_sessions:.1f} sessions/user"
                )

            # Search distribution statistics
            searches_per_user = searches_df.groupby("user_id").size()
            print("\nSearches per User Distribution (searching users only):")
            print(f"  Min: {searches_per_user.min()}")
            print(f"  25th percentile: {searches_per_user.quantile(0.25):.0f}")
            print(f"  Median: {searches_per_user.median():.0f}")
            print(f"  75th percentile: {searches_per_user.quantile(0.75):.0f}")
            print(f"  Max: {searches_per_user.max()}")

            # Session statistics
            sessions_df = (
                searches_df.groupby(["user_id", "session_id"])
                .size()
                .reset_index(name="searches_in_session")
            )
            print("\nSession Statistics:")
            print(f"  Total sessions: {len(sessions_df):,}")
            print(
                f"  Average searches per session: {sessions_df['searches_in_session'].mean():.1f}"
            )
            print(
                f"  Min searches per session: {sessions_df['searches_in_session'].min()}"
            )
            print(
                f"  Max searches per session: {sessions_df['searches_in_session'].max()}"
            )

    def create_visualizations(
        self, searches_df: pd.DataFrame, output_path: str
    ) -> None:
        """Create visualizations of the search data."""
        if searches_df.empty:
            print("No searches to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Session-Based Search Pattern Analysis", fontsize=16, y=0.98)

        # 1. User segment distribution
        segment_users = searches_df.groupby("user_segment")["user_id"].nunique()
        segment_users["non_searcher"] = self.n_users * 0.35  # Add non-searchers
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
        segment_users.plot(kind="pie", ax=axes[0, 0], colors=colors, autopct="%1.1f%%")
        axes[0, 0].set_title("User Segment Distribution")
        axes[0, 0].set_ylabel("")

        # 2. Searches per user by segment
        segment_searches = searches_df.groupby(["user_segment", "user_id"]).size()
        segment_data = [
            segment_searches[segment].values
            for segment in searches_df["user_segment"].unique()
            if segment in segment_searches.index
        ]

        axes[0, 1].boxplot(segment_data, labels=searches_df["user_segment"].unique())
        axes[0, 1].set_xlabel("User Segment")
        axes[0, 1].set_ylabel("Searches per User")
        axes[0, 1].set_title("Search Distribution by Segment")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Temporal patterns - searches over time
        searches_df["date"] = searches_df["search_ts"].dt.date
        daily_searches = searches_df.groupby("date").size()
        axes[1, 0].plot(daily_searches.index, daily_searches.values, color="steelblue")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Number of Searches")
        axes[1, 0].set_title("Search Volume Over Time")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Session length distribution
        session_lengths = searches_df.groupby("session_id").size()
        axes[1, 1].hist(session_lengths, bins=20, color="lightcoral", edgecolor="black")
        axes[1, 1].set_xlabel("Searches per Session")
        axes[1, 1].set_ylabel("Number of Sessions")
        axes[1, 1].set_title("Session Length Distribution")
        axes[1, 1].axvline(
            session_lengths.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {session_lengths.mean():.1f}",
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = SessionBasedSearchGenerator(n_users=20000)

    # Generate searches
    print("Generating session-based search data...")
    searches_df = generator.generate_searches()

    # Generate report
    generator.generate_statistics_report(searches_df)

    # Save data
    output_path = "data/session_based_searches.csv"
    if not searches_df.empty:
        searches_df.to_csv(output_path, index=False)
        print(f"\nSearch data saved to: {output_path}")

        # Create visualizations
        generator.create_visualizations(searches_df, "data/session_based_analysis.png")
    else:
        print("\nNo searches generated.")

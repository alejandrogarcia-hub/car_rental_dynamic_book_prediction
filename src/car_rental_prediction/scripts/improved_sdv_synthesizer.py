"""
Improved SDV synthesizer that preserves temporal distributions better.

This module provides enhanced synthesis capabilities for time-series data,
particularly for preserving hourly, daily, and monthly patterns in search timestamps.
"""

import warnings

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

warnings.filterwarnings("ignore")


class TemporalAwareSearchSynthesizer:
    """
    Enhanced synthesizer for search data that preserves temporal patterns.

    This synthesizer:
    1. Extracts temporal features before synthesis
    2. Uses advanced SDV models for better distribution matching
    3. Reconstructs timestamps from synthetic temporal features
    """

    def __init__(self, synthesizer_type="ctgan", seed=2025):
        """
        Initialize the temporal-aware synthesizer.

        Args:
            synthesizer_type: Type of synthesizer ('ctgan', 'tvae', or 'gaussian')
            seed: Random seed for reproducibility
        """
        self.synthesizer_type = synthesizer_type
        self.seed = seed
        self.synthesizer = None
        self.metadata = None
        self.temporal_distributions = {}

    def _extract_temporal_features(self, df):
        """
        Extract temporal features from search_ts column.

        Returns DataFrame with additional temporal columns for better synthesis.
        """
        df = df.copy()

        # Convert to datetime
        df["search_ts"] = pd.to_datetime(df["search_ts"])

        # Extract temporal components
        df["year"] = df["search_ts"].dt.year
        df["month"] = df["search_ts"].dt.month
        df["day"] = df["search_ts"].dt.day
        df["hour"] = df["search_ts"].dt.hour
        df["minute"] = df["search_ts"].dt.minute
        df["second"] = df["search_ts"].dt.second
        df["dayofweek"] = df["search_ts"].dt.dayofweek
        df["dayofyear"] = df["search_ts"].dt.dayofyear

        # Add cyclical features for better pattern capture
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        # Store original temporal distributions
        self.temporal_distributions = {
            "hour": df["hour"].value_counts(normalize=True).sort_index(),
            "dayofweek": df["dayofweek"].value_counts(normalize=True).sort_index(),
            "month": df["month"].value_counts(normalize=True).sort_index(),
        }

        # Drop the original timestamp column for synthesis
        # We drop it because we've extracted all temporal features from it
        # and will reconstruct it later from those features
        df_for_synthesis = df.drop("search_ts", axis=1)

        # Also drop session_id if present - it's a complex string that causes issues
        if "session_id" in df_for_synthesis.columns:
            df_for_synthesis = df_for_synthesis.drop("session_id", axis=1)
            self.has_session_id = True
        else:
            self.has_session_id = False

        # Get min/max dates as pandas Timestamps
        min_date = pd.to_datetime(df["search_ts"]).min()
        max_date = pd.to_datetime(df["search_ts"]).max()

        return df_for_synthesis, min_date, max_date

    def _reconstruct_timestamps(self, synthetic_df, min_date, max_date):
        """
        Reconstruct search_ts from synthetic temporal features.

        Uses the synthesized temporal components to create realistic timestamps.
        """
        synthetic_df = synthetic_df.copy()

        # Ensure integer values for date components
        synthetic_df["year"] = (
            synthetic_df["year"].clip(min_date.year, max_date.year).astype(int)
        )
        synthetic_df["month"] = synthetic_df["month"].clip(1, 12).astype(int)
        synthetic_df["day"] = synthetic_df["day"].clip(1, 31).astype(int)
        synthetic_df["hour"] = synthetic_df["hour"].clip(0, 23).astype(int)
        synthetic_df["minute"] = synthetic_df["minute"].clip(0, 59).astype(int)
        synthetic_df["second"] = synthetic_df["second"].clip(0, 59).astype(int)

        # Create timestamps
        timestamps = []
        for _, row in synthetic_df.iterrows():
            try:
                # Handle invalid dates (e.g., Feb 31)
                ts = pd.Timestamp(
                    year=int(row["year"]),
                    month=int(row["month"]),
                    day=min(int(row["day"]), 28),  # Safe day to avoid month-end issues
                    hour=int(row["hour"]),
                    minute=int(row["minute"]),
                    second=int(row["second"]),
                )
                # Adjust day after creating timestamp
                max_day = pd.Timestamp(
                    year=ts.year, month=ts.month, day=1
                ).days_in_month
                final_day = min(int(row["day"]), max_day)
                ts = ts.replace(day=final_day)

            except Exception as _:
                # Fallback to a random timestamp within range
                ts = pd.Timestamp(min_date) + pd.Timedelta(
                    seconds=np.random.randint(
                        0, int((max_date - min_date).total_seconds())
                    )
                )

            timestamps.append(ts)

        synthetic_df["search_ts"] = timestamps

        # Drop the temporal feature columns
        columns_to_drop = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "dayofyear",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
        ]
        synthetic_df = synthetic_df.drop(
            columns=[col for col in columns_to_drop if col in synthetic_df.columns],
            axis=1,
        )

        # Regenerate session_id if it was present in original data
        if hasattr(self, "has_session_id") and self.has_session_id:
            # Create session IDs based on user_id and timestamp
            if "user_id" in synthetic_df.columns:
                synthetic_df["session_id"] = synthetic_df.apply(
                    lambda row: f"{row['user_id']}_{pd.to_datetime(row['search_ts']).strftime('%Y%m%d%H%M')}",
                    axis=1,
                )
            else:
                # Fallback if no user_id
                synthetic_df["session_id"] = [
                    f"session_{i}" for i in range(len(synthetic_df))
                ]

        return synthetic_df

    def _post_process_distribution(self, synthetic_df):
        """
        Post-process to better match original temporal distributions.

        This resamples timestamps to better match the original hourly distribution.
        """
        if "hour" not in self.temporal_distributions:
            return synthetic_df

        # Extract current hour distribution
        synthetic_df["temp_hour"] = pd.to_datetime(synthetic_df["search_ts"]).dt.hour

        # Calculate resampling weights based on target distribution
        current_hour_dist = synthetic_df["temp_hour"].value_counts(normalize=True)
        target_hour_dist = self.temporal_distributions["hour"]

        # Create resampling weights for each row
        weights = synthetic_df["temp_hour"].map(
            lambda h: target_hour_dist.get(h, 0.01)
            / max(current_hour_dist.get(h, 0.01), 0.001)
        )

        # Normalize weights
        weights = weights / weights.sum()

        # Resample based on weights
        n_samples = len(synthetic_df)
        resampled_indices = np.random.choice(
            synthetic_df.index, size=n_samples, replace=True, p=weights.values
        )

        synthetic_df = synthetic_df.loc[resampled_indices].reset_index(drop=True)
        synthetic_df = synthetic_df.drop("temp_hour", axis=1)

        # Regenerate IDs to maintain uniqueness
        if "search_id" in synthetic_df.columns:
            synthetic_df["search_id"] = range(1, len(synthetic_df) + 1)

        return synthetic_df

    def fit(self, data):
        """
        Fit the synthesizer to the search data.

        Args:
            data: DataFrame containing search data with search_ts column
        """
        # Extract temporal features
        data_with_features, self.min_date, self.max_date = (
            self._extract_temporal_features(data)
        )

        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data_with_features)

        # Update metadata for better synthesis
        # Set appropriate sdtypes for temporal features
        temporal_columns = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "dayofyear",
        ]
        for col in temporal_columns:
            if col in data_with_features.columns:
                self.metadata.update_column(col, sdtype="numerical")

        # Create synthesizer based on type
        if self.synthesizer_type == "ctgan":
            self.synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=100,  # Reduced for faster testing, use 300+ for production
                batch_size=500,
                verbose=False,
            )
        elif self.synthesizer_type == "tvae":
            self.synthesizer = TVAESynthesizer(
                metadata=self.metadata, epochs=300, batch_size=500, verbose=False
            )
        else:  # gaussian
            self.synthesizer = GaussianCopulaSynthesizer(
                metadata=self.metadata,
                default_distribution="beta",  # Better for bounded data
            )

        # Fit the synthesizer
        print(f"Fitting {self.synthesizer_type.upper()} synthesizer...")
        self.synthesizer.fit(data_with_features)
        print("✅ Synthesizer fitted successfully")

    def sample(self, n_samples):
        """
        Generate synthetic search data.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            DataFrame with synthetic search data including reconstructed timestamps
        """
        # Generate synthetic data with temporal features
        print(f"Generating {n_samples:,} synthetic samples...")
        synthetic_with_features = self.synthesizer.sample(num_rows=n_samples)

        # Reconstruct timestamps
        print("Reconstructing timestamps from temporal features...")
        synthetic_df = self._reconstruct_timestamps(
            synthetic_with_features, self.min_date, self.max_date
        )

        # Post-process to match distributions
        print("Post-processing to match temporal distributions...")
        synthetic_df = self._post_process_distribution(synthetic_df)

        # Sort by timestamp
        synthetic_df = synthetic_df.sort_values("search_ts").reset_index(drop=True)

        # Regenerate search_id to maintain order
        if "search_id" in synthetic_df.columns:
            synthetic_df["search_id"] = range(1, len(synthetic_df) + 1)

        print("✅ Synthetic data generated successfully")
        return synthetic_df

    def evaluate_quality(self, original_df, synthetic_df):
        """
        Evaluate the quality of temporal distribution preservation.

        Returns a dictionary with quality metrics.
        """
        # Convert timestamps
        original_df["search_ts"] = pd.to_datetime(original_df["search_ts"])
        synthetic_df["search_ts"] = pd.to_datetime(synthetic_df["search_ts"])

        # Extract temporal features
        orig_hours = original_df["search_ts"].dt.hour
        synth_hours = synthetic_df["search_ts"].dt.hour

        orig_dow = original_df["search_ts"].dt.dayofweek
        synth_dow = synthetic_df["search_ts"].dt.dayofweek

        orig_months = original_df["search_ts"].dt.month
        synth_months = synthetic_df["search_ts"].dt.month

        # Calculate distribution differences
        hour_dist_diff = np.abs(
            orig_hours.value_counts(normalize=True).sort_index()
            - synth_hours.value_counts(normalize=True).sort_index()
        ).mean()

        dow_dist_diff = np.abs(
            orig_dow.value_counts(normalize=True).sort_index()
            - synth_dow.value_counts(normalize=True).sort_index()
        ).mean()

        month_dist_diff = np.abs(
            orig_months.value_counts(normalize=True).sort_index()
            - synth_months.value_counts(normalize=True).sort_index()
        ).mean()

        quality_metrics = {
            "hour_distribution_difference": hour_dist_diff,
            "dayofweek_distribution_difference": dow_dist_diff,
            "month_distribution_difference": month_dist_diff,
            "overall_temporal_quality": 1
            - np.mean([hour_dist_diff, dow_dist_diff, month_dist_diff]),
        }

        return quality_metrics


def synthesize_searches_with_temporal_preservation(
    original_searches_df, synthesizer_type="ctgan", n_samples=None, seed=2025
):
    """
    Convenience function to synthesize search data with temporal preservation.

    Args:
        original_searches_df: Original search data
        synthesizer_type: Type of synthesizer to use
        n_samples: Number of samples (defaults to original size)
        seed: Random seed

    Returns:
        Synthetic search DataFrame
    """
    if n_samples is None:
        n_samples = len(original_searches_df)

    # Create and fit synthesizer
    synthesizer = TemporalAwareSearchSynthesizer(
        synthesizer_type=synthesizer_type, seed=seed
    )
    synthesizer.fit(original_searches_df)

    # Generate synthetic data
    synthetic_searches = synthesizer.sample(n_samples)

    # Evaluate quality
    quality = synthesizer.evaluate_quality(original_searches_df, synthetic_searches)
    print("\nTemporal Quality Metrics:")
    for metric, value in quality.items():
        print(f"  {metric}: {value:.4f}")

    return synthetic_searches


if __name__ == "__main__":
    # Example usage
    print("Loading sample search data...")
    sample_searches = pd.read_csv("../../../data/sample/searches.csv")

    # Use 50% of data for testing (still substantial)
    sample_size = int(len(sample_searches) * 0.5)
    sample_searches = sample_searches.sample(n=sample_size, random_state=42)

    print(f"Original data shape: {sample_searches.shape}")
    print("\nOriginal hourly distribution:")
    orig_hour_dist = (
        pd.to_datetime(sample_searches["search_ts"])
        .dt.hour.value_counts(normalize=True)
        .sort_index()
    )
    print(orig_hour_dist.head(10))

    print("\n" + "=" * 60)
    print("NOTE: CTGAN training may take 5-10 minutes for quality results")
    print("=" * 60 + "\n")

    # Generate synthetic data using CTGAN for best quality
    synthetic_searches = synthesize_searches_with_temporal_preservation(
        sample_searches,
        synthesizer_type="ctgan",  # Best quality for temporal patterns
        n_samples=len(sample_searches),  # Generate same size as input
        seed=2025,
    )

    print(f"\nSynthetic data shape: {synthetic_searches.shape}")
    print("\nSynthetic hourly distribution:")
    synth_hour_dist = (
        pd.to_datetime(synthetic_searches["search_ts"])
        .dt.hour.value_counts(normalize=True)
        .sort_index()
    )
    print(synth_hour_dist.head(10))

    # Show improvement
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON")
    print("=" * 60)
    print("\nMidnight (0:00) searches:")
    print(f"  Original: {orig_hour_dist.get(0, 0):.2%}")
    print(f"  Synthetic: {synth_hour_dist.get(0, 0):.2%}")
    print(
        f"  Difference: {abs(orig_hour_dist.get(0, 0) - synth_hour_dist.get(0, 0)):.2%}"
    )

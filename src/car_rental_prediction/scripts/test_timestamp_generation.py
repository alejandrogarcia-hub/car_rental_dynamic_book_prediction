"""
Test script to verify that search timestamps have realistic hourly distributions.
"""

import matplotlib.pyplot as plt
from realistic_search_generator import RealisticSearchGenerator

# Generate a small sample to test
print("Generating test search data...")
generator = RealisticSearchGenerator(
    n_users=1000, start_date="2024-01-01", end_date="2024-01-31", seed=42
)

searches_df = generator.generate_searches()

# Check timestamp distribution
print("\nTimestamp Analysis:")
print(f"Total searches: {len(searches_df)}")
print("Sample timestamps:")
for i in range(min(10, len(searches_df))):
    ts = searches_df.iloc[i]["search_ts"]
    print(f"  {ts}")

# Analyze hourly distribution
searches_df["hour"] = searches_df["search_ts"].dt.hour
hour_counts = searches_df["hour"].value_counts().sort_index()

print("\nHourly Distribution:")
for hour in range(24):
    count = hour_counts.get(hour, 0)
    pct = count / len(searches_df) * 100
    bar = "█" * int(pct * 2)  # Visual bar
    print(f"  {hour:02d}:00 - {count:4d} ({pct:5.2f}%) {bar}")

# Check for midnight-only issue
midnight_count = (searches_df["search_ts"].dt.hour == 0).sum()
midnight_pct = midnight_count / len(searches_df) * 100

print(f"\nMidnight searches: {midnight_count} ({midnight_pct:.1f}%)")
if midnight_pct > 50:
    print("⚠️  WARNING: Too many searches at midnight!")
else:
    print("✅ Hourly distribution looks realistic")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Hourly distribution
hour_counts.plot(kind="bar", ax=ax1, color="steelblue")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Number of Searches")
ax1.set_title("Search Distribution by Hour")
ax1.grid(axis="y", alpha=0.3)

# Expected vs actual comparison
expected_weights = [
    0.01,
    0.005,
    0.005,
    0.005,
    0.005,
    0.01,  # 0-5 AM
    0.02,
    0.04,
    0.06,
    0.07,
    0.075,
    0.08,  # 6-11 AM
    0.09,
    0.085,
    0.075,
    0.07,
    0.065,
    0.08,  # 12-5 PM
    0.09,
    0.095,
    0.09,
    0.07,
    0.04,
    0.02,  # 6-11 PM
]
expected_weights = [w / sum(expected_weights) for w in expected_weights]

actual_weights = [hour_counts.get(h, 0) / len(searches_df) for h in range(24)]

hours = list(range(24))
ax2.plot(hours, expected_weights, "o-", label="Expected", linewidth=2)
ax2.plot(hours, actual_weights, "s-", label="Actual", linewidth=2)
ax2.set_xlabel("Hour of Day")
ax2.set_ylabel("Proportion of Searches")
ax2.set_title("Expected vs Actual Hourly Distribution")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("timestamp_distribution_test.png", dpi=150)
print("\nVisualization saved to: timestamp_distribution_test.png")

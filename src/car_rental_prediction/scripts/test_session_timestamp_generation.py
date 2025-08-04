"""
Test script to verify that session-based search timestamps have realistic hourly distributions.
"""

from session_based_search_generator import SessionBasedSearchGenerator

# Generate a small sample to test
print("Generating test search data with session-based generator...")
generator = SessionBasedSearchGenerator(
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

# Check that searches within sessions have proper timestamps
print("\nSession timestamp analysis:")
sample_sessions = searches_df["session_id"].unique()[:5]
for session_id in sample_sessions:
    session_searches = searches_df[searches_df["session_id"] == session_id]
    print(f"\nSession {session_id}:")
    for _, search in session_searches.iterrows():
        print(f"  {search['search_ts']}")

print("\n✅ Session-based timestamp generation test completed")

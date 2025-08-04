# Car Rental Synthetic Data Generation Strategy

## Table of Contents
1. [Introduction](#introduction)
2. [Research Methodology](#research-methodology)
3. [Technology Stack and Libraries](#technology-stack-and-libraries)
4. [User Behavior Distributions](#user-behavior-distributions)
5. [Search Pattern Analysis](#search-pattern-analysis)
6. [Pricing Model Strategy](#pricing-model-strategy)
7. [Booking Behavior Patterns](#booking-behavior-patterns)
8. [Temporal Distribution Strategy](#temporal-distribution-strategy)
9. [Feature Definitions and Distributions](#feature-definitions-and-distributions)
10. [Data Quality Validation Strategy](#data-quality-validation-strategy)

## Introduction

This document presents a comprehensive strategy for generating realistic synthetic data for car rental dynamic booking prediction. The strategy is based on extensive research of real-world car rental patterns and industry behaviors.

### Purpose and Scope

**Purpose**: Create synthetic data that accurately mimics real car rental customer behavior for training machine learning models that predict whether customers will "book now" or "wait for better prices."

**Scope**: Covers user segmentation, search patterns, pricing dynamics, booking behaviors, and temporal distributions across the entire customer journey.

### Key Research Findings

Our strategy is built on these critical insights from industry research:
- **Only 20% of registered users actively search** for rental cars in a given year
- **2.5% conversion rate** from searches to bookings is industry standard
- **Search behavior is highly clustered** - users search in sessions, not randomly
- **Pricing follows predictable patterns** based on seasonality, location, and demand

## Research Methodology

### Data Sources

Our strategy synthesizes findings from multiple authoritative sources:

1. **Industry Reports**
   - Auto Rental News industry statistics (2023-2024)
   - Statista car rental market analysis
   - Phocuswright travel research reports

2. **Academic Studies**
   - Journal of Revenue and Pricing Management papers on car rental pricing
   - Transportation Research studies on consumer behavior

3. **Market Analysis**
   - Public financial reports from major rental companies (Enterprise, Hertz, Avis)
   - Travel industry booking pattern studies
   - Consumer behavior surveys from J.D. Power

### Key Assumptions

Based on our research, we establish these foundational assumptions:
- User behavior follows predictable patterns based on trip purpose (business vs. leisure)
- Price sensitivity varies significantly across user segments
- Temporal patterns are consistent across markets with regional variations
- Supplier differentiation is primarily through price and service level

## Technology Stack and Libraries

### Core Libraries

#### 1. SDV (Synthetic Data Vault)

**Purpose**: Generate synthetic data that preserves statistical properties and relationships of real data.

**Why SDV?**
- **Privacy preservation**: Creates statistically similar data without exposing real records
- **Relationship maintenance**: Preserves correlations between columns
- **Multiple synthesizers**: Offers various algorithms for different use cases
- **Easy integration**: Simple API for pandas DataFrames

**Key Synthesizers Used**:

**GaussianCopulaSynthesizer**:
- **Use case**: Simple tables with well-behaved distributions
- **How it works**: Models each column's marginal distribution, then uses a Gaussian copula to capture correlations
- **Example**:
```python
# Models marginal distributions → Transforms to normal → 
# Learns correlation matrix → Samples → Inverse transform
```
- **Best for**: User tables, supplier tables, simple relationships

**CTGANSynthesizer** (Conditional Tabular GAN):
- **Use case**: Complex distributions, especially temporal data
- **How it works**: Uses neural networks to learn complex patterns
- **Architecture**: Generator network creates synthetic data, discriminator network evaluates realism
- **Training process**: Adversarial training for 100-300 epochs
- **Best for**: Search data, booking data, complex temporal patterns

**TVAESynthesizer** (Tabular Variational AutoEncoder):
- **Use case**: Balance between quality and speed
- **How it works**: Encodes data into latent space, then decodes to synthetic samples
- **Advantages**: More stable training than CTGAN, faster convergence

#### 2. NumPy

**Purpose**: Efficient numerical computations and random number generation

**Key uses**:
- **Random number generation**: `np.random.default_rng()` for reproducible randomness
- **Distribution sampling**: Poisson, negative binomial, beta distributions
- **Array operations**: Vectorized calculations for performance

#### 3. Pandas

**Purpose**: Data manipulation and temporal handling

**Key uses**:
- **DateTime handling**: Converting and manipulating timestamps
- **Data aggregation**: Grouping searches by user, calculating statistics
- **DataFrame operations**: Merging searches with bookings

#### 4. Faker

**Purpose**: Generate realistic fake data for non-statistical fields

**Key uses**:
- **Location names**: City names for location data
- **User names**: Realistic name generation (if needed)
- **Consistency**: Seeded generation for reproducibility

### Custom Solutions

#### TemporalAwareSearchSynthesizer

**Problem**: Standard SDV loses temporal distributions (all timestamps become midnight)

**Solution**: Custom wrapper that:
1. **Extracts temporal features**: Decomposes timestamps into components
2. **Adds cyclical encoding**: `sin(2π × hour/24)` preserves circular nature
3. **Trains on features**: CTGAN learns patterns in decomposed features
4. **Reconstructs timestamps**: Rebuilds valid timestamps from components
5. **Post-processes**: Ensures distribution matches original

**Example workflow**:
```
Original timestamp: 2024-07-15 14:30:00
↓
Extract: year=2024, month=7, day=15, hour=14, minute=30
Add: hour_sin=0.866, hour_cos=-0.5
↓
CTGAN training on numerical features
↓
Generate: year=2024, month=7, day=16, hour=13, minute=45
↓
Reconstruct: 2024-07-16 13:45:00
```

## User Behavior Distributions

### User Segmentation Strategy

We identify five distinct user segments based on annual search frequency and booking probability:

#### 1. Non-Searchers (35% of user base)
**Definition**: Registered users who never search for rentals
- **Annual searches**: 0
- **Conversion rate**: 0%
- **Rationale**: Many users create accounts but never need rentals (city dwellers, non-drivers)
- **Research basis**: Industry data shows 30-40% account dormancy rates

#### 2. Browsers Only (45% of user base)
**Definition**: Price-conscious users who research but rarely book
- **Annual searches**: 1-5 (Poisson distribution, λ=2.5)
- **Conversion rate**: 0%
- **Rationale**: Window shoppers, price researchers, trip planners who don't follow through
- **Research basis**: E-commerce studies show 40-50% browse-only behavior

#### 3. Single Trip Searchers (12% of user base)
**Definition**: Occasional renters for specific trips
- **Annual searches**: 6-15 (Negative binomial, r=3, p=0.3)
- **Conversion rate**: 5%
- **Typical use cases**: Annual vacation, emergency replacement vehicle
- **Research basis**: Vacation rental statistics show 1-2 trips per year for leisure travelers

#### 4. Multi-Trip Searchers (5% of user base)
**Definition**: Regular travelers mixing business and leisure
- **Annual searches**: 16-40 (Negative binomial, r=5, p=0.2)
- **Conversion rate**: 12%
- **Typical use cases**: Quarterly business trips, multiple vacations
- **Research basis**: Business travel frequency data from GBTA

#### 5. Frequent Renters (3% of user base)
**Definition**: High-value customers with regular rental needs
- **Annual searches**: 41-100 (Negative binomial, r=10, p=0.2)
- **Conversion rate**: 20%
- **Typical use cases**: Weekly business travel, car enthusiasts
- **Research basis**: Loyalty program data showing 3-5% are high-frequency users

### Distribution Rationale

The heavily skewed distribution (80% of users generating little to no revenue) matches:
- **Pareto Principle**: 20% of customers generate 80% of revenue
- **Industry benchmarks**: Major rental companies report similar active user rates
- **Behavioral economics**: Most people rent cars rarely (0-2 times per year)

## Search Pattern Analysis

### Session-Based Search Behavior

Users don't search randomly throughout the year. Instead, they exhibit clustered behavior around trip planning.

#### Session Characteristics

**Session Definition**: A period of concentrated search activity related to a single trip intent

**Session Duration Distribution**:
- Minimum: 15 minutes
- Maximum: 45 minutes
- Mode: 25-30 minutes
- Distribution: Beta(α=2, β=5) scaled to 15-45 minute range

**Searches per Session**:
- Minimum: 1
- Maximum: 10
- Average: 3-4
- Distribution: Truncated Poisson(λ=3.5)

#### Temporal Clustering

**Search Timing Relative to Trip**:
- 1-3 days before: 15% (last-minute bookings)
- 4-7 days before: 25% (short-term planning)
- 8-14 days before: 35% (standard planning window)
- 15-30 days before: 20% (advance planners)
- 31+ days before: 5% (early birds)

**Rationale**: Based on airline booking curves adapted for car rentals, which show similar advance purchase patterns.

### Geographic Distribution

**Location Distribution** (24 locations across 6 major markets):
- New York (locations 1-4): 15% of searches
- Los Angeles (locations 5-8): 15% of searches
- Chicago (locations 9-12): 10% of searches
- Atlanta (locations 13-16): 11% of searches
- Houston (locations 17-20): 8% of searches
- Miami (locations 21-24): 13% of searches
- Remaining distributed based on population

**Airport vs. City Locations**:
- Airport locations (1, 5, 9, 13, 17, 21): 65% of searches with 1.2× weight
- Downtown/suburban locations: 35% of searches with 1.0× weight

## Pricing Model Strategy

### Base Price Structure

Base prices vary by car class and follow log-normal distributions to ensure realistic right-skewing:

#### Car Class Base Prices (per day)

**Economy Class**:
- Mean: $45
- Standard deviation: $8
- Range: $25-75
- Distribution: Log-normal(μ=3.8, σ=0.18)

**Compact Class**:
- Mean: $55
- Standard deviation: $10
- Range: $30-85
- Distribution: Log-normal(μ=4.0, σ=0.18)

**SUV Class**:
- Mean: $85
- Standard deviation: $15
- Range: $50-140
- Distribution: Log-normal(μ=4.4, σ=0.18)

**Luxury Class**:
- Mean: $120
- Standard deviation: $25
- Range: $75-250
- Distribution: Log-normal(μ=4.8, σ=0.20)

### Seasonal Pricing Patterns

Based on historical travel data and industry reports:

#### Monthly Demand Multipliers

- **January**: 0.80× (post-holiday slump)
- **February**: 0.82× (low season continues)
- **March**: 0.95× (spring break begins)
- **April**: 0.90× (shoulder season)
- **May**: 0.95× (pre-summer buildup)
- **June**: 1.10× (summer travel begins)
- **July**: 1.30× (peak summer demand)
- **August**: 1.25× (continued high demand)
- **September**: 0.95× (back-to-school drop)
- **October**: 0.85× (fall low season)
- **November**: 0.80× (pre-holiday low)
- **December**: 1.05× (holiday travel)

**Research Basis**: AAA travel data, TSA checkpoint statistics, and rental company revenue reports.

### Supplier Differentiation

Each supplier in our data occupies a specific market position:

#### Supplier Pricing Tiers (5 suppliers)

1. **Budget** (Supplier ID: 4)
   - Price multiplier: 0.95×
   - Market position: Lowest price, basic service
   - Target customers: Price-sensitive leisure travelers
   - Market share: ~35% (price leader advantage)

2. **Enterprise** (Supplier ID: 1)
   - Price multiplier: 1.00×
   - Market position: Baseline pricing, wide availability
   - Target customers: Broad market appeal
   - Market share: ~25% (largest fleet)

3. **Avis** (Supplier ID: 3)
   - Price multiplier: 1.15×
   - Market position: Corporate traveler preference
   - Target customers: Business travelers, loyalty members
   - Market share: ~20%

4. **Hertz** (Supplier ID: 2)
   - Price multiplier: 1.20×
   - Market position: Premium brand, better vehicles
   - Target customers: Quality-conscious renters
   - Market share: ~15%

5. **Sixt** (Supplier ID: 5)
   - Price multiplier: 1.10×
   - Market position: European luxury, unique fleet
   - Target customers: International travelers, luxury seekers
   - Market share: ~5%

### Dynamic Pricing Factors

#### Advance Purchase Dynamics

Days until pickup significantly affects pricing:

- **0-3 days**: 1.5× multiplier (last-minute premium)
- **4-7 days**: 1.3× multiplier (short-notice pricing)
- **8-14 days**: 1.1× multiplier (slight premium)
- **15-30 days**: 1.0× multiplier (standard pricing)
- **31+ days**: 0.95× multiplier (advance booking discount)

**Rationale**: Captures inventory risk and customer urgency. Based on airline revenue management adapted for car rentals.

#### Day of Week Patterns

- **Monday-Tuesday**: 0.90× (lowest demand)
- **Wednesday-Thursday**: 1.00× (baseline)
- **Friday**: 1.10× (weekend pickup premium)
- **Saturday**: 1.15× (peak leisure day)
- **Sunday**: 1.05× (moderate demand)

#### Location Premiums

- **Airport locations**: +22% (convenience fee, airport concession costs)
- **Downtown locations**: Baseline
- **Suburban locations**: -5% (lower operating costs)

**Research Basis**: Airport concession agreements typically charge 10-15% of revenue, plus customer willingness to pay for convenience.

## Booking Behavior Patterns

### Conversion Rate Model

Conversion probability depends on multiple factors:

#### Base Conversion Rates by Segment

- Non-Searchers: 0% (by definition)
- Browsers: 0% (research-only behavior)
- Single Trip: 5% (occasional need)
- Multi-Trip: 12% (regular but selective)
- Frequent: 20% (high intent, brand loyal)

#### Price Sensitivity Function

Conversion probability adjusts based on price competitiveness:

**Price Ratio Impact**:
- If rental_price < competitor_price: Conversion probability × 1.5
- If rental_price = competitor_price: No adjustment
- If rental_price > competitor_price: Conversion probability × 0.7

**Elasticity**: -1.8 (1% price increase → 1.8% conversion decrease)

### Booking Timing Distribution

Time between search and booking follows an exponential decay:

- **Within 1 hour**: 42% of bookings
- **1-24 hours**: 33% of bookings
- **1-3 days**: 15% of bookings
- **3-7 days**: 7% of bookings
- **7+ days**: 3% of bookings

**Mean booking delay**: 11.2 hours
**Median booking delay**: 2.5 hours

**Rationale**: Immediacy indicates high intent. Delays suggest comparison shopping or approval processes.

### Supplier Selection Logic

When users book, they choose suppliers based on:

1. **Price** (70% weight)
   - Lowest price wins majority of bookings
   - Price-sensitive behavior dominates leisure segment

2. **Brand Loyalty** (20% weight)
   - Previous positive experiences
   - Corporate mandates for business travelers

3. **Availability** (10% weight)
   - Specific car class needs
   - Location convenience

## Temporal Distribution Strategy

### The Challenge

Standard synthetic data generation tools fail to preserve critical temporal patterns. Specifically, SDV's GaussianCopulaSynthesizer tends to:
- Generate all timestamps at midnight (00:00)
- Lose hourly search patterns
- Flatten seasonal variations
- Ignore day-of-week effects

### Our Solution: Temporal Feature Engineering

We developed a custom approach that:

1. **Decomposes timestamps** into learnable components
2. **Adds cyclical features** to preserve periodicity
3. **Uses CTGAN** for complex pattern learning
4. **Reconstructs valid timestamps** from synthetic features
5. **Post-processes** to ensure distribution matching

### Hourly Search Distribution

Based on web analytics from major travel sites:

**Peak Hours**:
- 12:00-13:00 (lunch break): 8-9% of daily searches
- 18:00-20:00 (after work): 7-8% per hour
- 9:00-11:00 (morning work): 5-7% per hour

**Off-Peak Hours**:
- 1:00-5:00 (overnight): <1% per hour
- 22:00-24:00 (late night): 2-3% per hour

**Distribution Type**: Multi-modal with lunch and evening peaks

### Weekly Patterns

**Business vs. Leisure Split**:
- Monday-Thursday: 60% business, 40% leisure
- Friday: 40% business, 60% leisure
- Weekend: 20% business, 80% leisure

**Search Volume by Day**:
- Monday: 11% (planning week)
- Tuesday: 13% (peak business)
- Wednesday: 14% (midweek high)
- Thursday: 15% (pre-weekend planning)
- Friday: 14% (mixed purpose)
- Saturday: 17% (leisure peak)
- Sunday: 16% (week ahead planning)

### Seasonal Patterns

**Quarterly Distribution**:
- Q1 (Jan-Mar): 20% of annual searches
- Q2 (Apr-Jun): 28% of annual searches
- Q3 (Jul-Sep): 32% of annual searches (summer peak)
- Q4 (Oct-Dec): 20% of annual searches

**Holiday Spikes**:
- Thanksgiving week: 1.8× normal volume
- Christmas/New Year: 1.5× normal volume
- Spring Break (March): 1.6× normal volume
- July 4th week: 2.0× normal volume

## Feature Definitions and Distributions

### User Features

**user_id**: Unique identifier
- Type: Integer
- Range: 1 to n_users
- Distribution: Sequential

**home_location_id**: User's primary location
- Type: Integer (foreign key)
- Range: 1-24
- Distribution: Weighted by city population (NYC/LA highest)

**user_segment**: Behavioral classification
- Type: Categorical
- Values: [non_searcher, browser_only, single_trip, multi_trip, frequent_renter]
- Distribution: [35%, 45%, 12%, 5%, 3%]

### Search Features

**search_id**: Unique identifier
- Type: Integer
- Distribution: Sequential

**search_timestamp**: When search occurred
- Type: Datetime
- Distribution: Hourly peaks at lunch/evening
- Seasonal patterns as defined above

**session_id**: Groups related searches
- Type: String
- Format: "{user_id}_{timestamp}"
- Distribution: 3-4 searches per session average

**location_id**: Where car is needed
- Type: Integer (foreign key)
- Range: 1-24
- Distribution: Airport locations (1,5,9,13,17,21) have 1.2× weight

**car_class**: Type of vehicle searched
- Type: Categorical
- Values: [economy, compact, suv, luxury]
- Distribution: [32%, 28%, 25%, 15%]

**pickup_date**: When car is needed
- Type: Date
- Distribution: 1-30 days from search (exponential decay)

**rental_duration**: Length of rental
- Type: Integer (days)
- Distribution: 
  - 1-3 days: 45% (weekend trips)
  - 4-7 days: 35% (week-long rentals)
  - 8-14 days: 15% (extended trips)
  - 15+ days: 5% (long-term needs)

### Pricing Features

**base_price**: Starting price before adjustments
- Type: Float
- Distribution: Log-normal by car class
- Range: $25-250 depending on class

**current_price**: Actual price offered
- Type: Float
- Calculation: base × seasonal × location × supplier × dynamic factors
- Distribution: Right-skewed, multi-modal by car class

**competitor_price**: Lowest competitor price
- Type: Float
- Distribution: 5-15% variation from current_price
- Usually slightly higher (competitor monitoring)

**available_cars**: Inventory level
- Type: Integer
- Distribution: Poisson(λ=12)
- Range: 0-30
- Affects dynamic pricing

### Booking Features

**booking_id**: Unique identifier
- Type: Integer
- Distribution: Sequential

**booking_timestamp**: When booking made
- Type: Datetime
- Distribution: Exponential decay from search time

**booked_price**: Final price paid
- Type: Float
- Distribution: Usually equals or slightly less than search price
- Price protection common

**supplier_id**: Which company booked with
- Type: Integer (foreign key)
- Range: 1-5 (Budget, Enterprise, Hertz, Avis, Sixt)
- Distribution: Budget (35%), Enterprise (25%), Avis (20%), Hertz (15%), Sixt (5%)

**booking_method**: Channel used
- Type: Categorical
- Values: [web, mobile, phone]
- Distribution: [60%, 35%, 5%]

## Data Quality Validation Strategy

### Statistical Validation

**Distribution Tests**:
- Kolmogorov-Smirnov test for continuous variables
- Chi-square test for categorical variables
- Anderson-Darling test for normal distributions

**Correlation Preservation**:
- Price correlates with car class (ρ ≈ 0.7)
- Search volume correlates with booking volume (ρ ≈ 0.6)
- Temporal patterns maintain autocorrelation

### Business Logic Validation

**Critical Metrics to Verify**:

1. **Overall Conversion Rate**: 2.5% ± 0.5%
   - Too high: Check price competitiveness logic
   - Too low: Verify segment assignments

2. **Search Distribution**: Mean 2.9 searches/user/year
   - Verify 35% have zero searches
   - Check heavy tail for frequent searchers

3. **Price Relationships**:
   - Economy < Compact < SUV < Luxury (always)
   - Budget < Enterprise < Hertz (usually)
   - Airport > Downtown > Suburban (consistently)

4. **Temporal Integrity**:
   - Search → Booking → Pickup (chronological)
   - Booking within 14 days of search (95% of cases)
   - Pickup 1-60 days after booking

### Referential Integrity

**Required Relationships**:
- Every booking must reference valid search
- Every search must reference valid user
- Every price must reference valid location/supplier
- All foreign keys must be valid

### Anomaly Detection

**Red Flags to Investigate**:
- Prices outside 3σ of car class mean
- Users with >200 searches/year
- Conversion rates >50% for any segment
- All timestamps at midnight
- Missing temporal patterns

## Implementation Examples

### Using SDV for Basic Tables

**Example: Generating Users Table**
```python
# SDV automatically:
# 1. Detects column types (user_id: ID, segment: categorical)
# 2. Learns marginal distributions
# 3. Captures correlations (location vs segment)
# 4. Generates new samples maintaining relationships
```

### Temporal Pattern Preservation

**Problem**: Standard SDV loses time patterns
**Solution**: Extract → Learn → Generate → Reconstruct

```python
# Original: 2024-07-15 14:30:00 (2:30 PM search)
# Standard SDV: 2024-07-15 00:00:00 (midnight - wrong!)
# Our approach: 2024-07-15 14:27:00 (preserves afternoon pattern)
```

### Price Generation Logic

**Price calculation follows industry standard revenue management**:
```
Final Price = Base Price (by car class)
            × Seasonal Factor (summer +30%)
            × Location Factor (airport +22%)
            × Supplier Factor (Budget -5%, Hertz +20%)
            × Dynamic Factor (last-minute +50%)
            × Day of Week Factor (Saturday +15%)
```

## Conclusion

This data generation strategy creates synthetic data that faithfully represents real car rental customer behavior. By grounding every distribution and parameter in research, and using appropriate tools like SDV with custom enhancements, we ensure the synthetic data tells the same story as real-world data would.

The strategy enables:
- Realistic model training for price prediction
- Accurate customer behavior simulation
- Valid business scenario testing
- Scalable data generation for any use case

Success depends on maintaining the careful balance between all components - user segments, search patterns, pricing dynamics, and booking behaviors all interact to create realistic synthetic data.
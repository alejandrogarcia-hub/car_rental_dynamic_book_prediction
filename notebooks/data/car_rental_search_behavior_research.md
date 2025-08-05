# Car Rental Search Behavior Research: Real-World Statistical Distributions

## Executive Summary

This document compiles research findings on real-world statistical distributions for car rental user search behavior patterns, location popularity, car class preferences, and seasonal patterns based on academic research and industry reports.

## 1. User Search Behavior Patterns

### Search Frequency Distribution

**Key Finding**: On average, consumers rent **2 cars per year**, with 74% primarily using rental cars for vacation and leisure travel (Grand View Research, 2024).

#### User Segmentation by Usage Frequency

Research identifies distinct user segments based on usage patterns:

1. **Regular vs Occasional Users**
   - **Occasional users**: Majority of users (specific percentage not provided)
   - **Regular users**: Minority segment with higher transaction frequency

2. **Detailed User Clusters** (Based on k-means clustering analysis)
   - **Short-term frequent heavy use pattern**
   - **Short-term frequent light use pattern**
   - **Long-term frequent light use pattern**
   - **Long-term occasional light use pattern**
   - **Long-term frequent heavy use pattern**

3. **User Loyalty Categories**
   - **Lost users**
   - **Early loyal users**
   - **Late loyal users**
   - **Motivated users**

### Temporal Usage Patterns

- **Daily distribution**: Noticeable morning and evening peaks during weekdays
- **Weekly patterns**: 8 types of typical weeks identified
  - Cluster C4 (occasional weekend use) is very popular among members
- **Search-to-booking conversion rate**: 2.5% (200,000 searches → 5,000 bookings)
  - Lower than general travel industry (4.7%) due to price comparison behavior

### Demographics

- **Gender distribution**:
  - Males: 50% of trips
  - Females: 37.64%
  - Organizations: 12.23%
  - Men have 4.3x greater likelihood of using rental cars than women

## 2. Location Popularity Distribution

### Regional Market Share (2024)

1. **North America**: 36.39% of global revenue
2. **Europe**: 24% of global revenue
3. **Asia-Pacific**: Fastest growing (CAGR 12.7% from 2025-2030)

### Popular Destinations

#### United States

- **Top states by rental company concentration**:
  - California: 1,355 companies
  - Florida: 1,050 companies
  - Texas: 960 companies
- **High-demand cities**: Los Angeles, New York City, Las Vegas, Miami

#### Europe

- **High-demand countries**: France, Italy, Spain
- **Popular regions**: Costa Brava (Spain), French Riviera

### Usage Type Distribution

- **Local usage**: 46.8% (US market)
- **Airport transport**: 38% (US market)
- **Leisure/tourism**: 62.39% (2022 data)

## 3. Car Class Preferences

### Vehicle Type Distribution

1. **Economy cars**: 32% global revenue share (29.8% in US)
   - US fleet: 2 million economy vehicles
2. **Compact cars**: Second most popular
3. **SUVs**: Growing segment
4. **Luxury vehicles**: Smallest segment

### ACRISS Classification System

The industry uses a 4-character code system:

- 1st character: Vehicle size/luxury factor
- 2nd character: Chassis type and doors
- 3rd character: Transmission type
- 4th character: Fuel/AC specifications

### Customer Selection Factors

Primary considerations:

1. Number of passengers
2. Luggage capacity
3. Trip duration
4. Budget constraints
5. Fuel economy

## 4. Seasonal and Pricing Patterns

### Seasonal Variations

- **Pricing**: Varies by vehicle size, seasonality, length of rental, availability, and location
- **Vehicle availability**: Certain types have strong seasonal patterns
  - Example: Convertibles not available in northern locations during winter

### Distribution Channels

- **Offline channels**: Growing from 61% (2017) to projected 72% (2027)
- **Online channels**: Expected to generate 73% of revenue by 2028

## 5. Market Statistics and Projections

### User Base Growth

- 2019: 450.5 million users globally
- 2020: 284.59 million (COVID-19 impact)
- 2023: 547.02 million
- 2028 projection: 644.4 million

### User Penetration

- 2023: 6.7% global penetration
- 2027 projection: 7.8%

### Average Revenue Per User (ARPU)

- Asia: $141.40
- Europe: $279.60
- United States: $630
- Global average: $181.00

## 6. Key Behavioral Insights for Modeling

### Search Behavior Patterns

1. **Search inputs commonly tracked**:
   - Locations (pickup/dropoff)
   - Vehicle types
   - Dates
   - Price ranges

2. **Browsing patterns**:
   - Time spent on specific car models
   - Filter usage patterns
   - Comparison shopping behavior

3. **Booking history indicators**:
   - Frequencies
   - Trip purposes
   - Vehicle preferences

### Operational Implications

- Peak rental/return periods at each station are predictable
- Vehicle dispatch can be optimized using demand patterns
- User segmentation enables personalized service offerings

## Data Sources and Limitations

### Primary Sources

- Academic papers from Journal of Advanced Transportation
- Market research from Statista, Grand View Research, Mordor Intelligence
- Industry reports from Auto Rental News, Enterprise Apps Today
- Consumer behavior studies from Zubie, NerdWallet

### Limitations

- Specific search frequency per user per year not explicitly documented
- Most data focuses on transactions rather than search behavior
- Regional variations may not be fully captured
- COVID-19 significantly impacted 2020-2021 patterns

## Recommendations for Synthetic Data Generation

Based on this research, the synthetic data generator should incorporate:

1. **User frequency distribution**: Poisson or negative binomial distribution with mean ~2 rentals/year
2. **Location popularity**: Weight by regional market share percentages
3. **Car class distribution**: Match the 32% economy, decreasing for higher classes
4. **Temporal patterns**: Include daily peaks and weekly patterns
5. **Conversion rates**: Apply 2.5% search-to-booking conversion
6. **Demographics**: Reflect gender and user type distributions
7. **Seasonal variations**: Implement location-based vehicle availability constraints

## Future Research Needs

1. Detailed search frequency distributions per user segment
2. Cross-regional price elasticity data
3. Mobile vs desktop conversion rate differences
4. Impact of loyalty programs on search behavior
5. Real-time pricing response patterns

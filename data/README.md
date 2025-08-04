# Data Directory

This directory contains the datasets used for model training and evaluation, organized according to MLOps best practices.

## Directory Structure

```
data/
├── raw/           # Original, immutable data files
├── processed/     # Cleaned and processed data ready for modeling
└── external/      # External data sources and references
```

## Data Files

### Processed Data (`processed/`)
Training and testing datasets generated from the synthetic data pipeline:

- `synthetic_users.csv` - User profiles and behavior segments
- `synthetic_searches.csv` - Search history with temporal patterns
- `synthetic_bookings.csv` - Booking records and conversions
- `synthetic_rental_prices.csv` - Price data with temporal dynamics
- `synthetic_competitor_prices.csv` - Competitive pricing information
- `synthetic_suppliers.csv` - Supplier information and configurations
- `synthetic_locations.csv` - Location data and weights
- `synthetic_car_classes.csv` - Car class definitions and pricing

## Data Quality

All datasets have been validated for:
- Referential integrity across tables
- Realistic temporal patterns
- Proper class distributions
- Business logic consistency

## Usage

Model training scripts expect data to be available in the `processed/` directory. The data pipeline generates approximately:
- 20,000 users across 5 behavioral segments
- 57,000+ searches with realistic patterns
- 3,200+ bookings (2.5% conversion rate)
- 8,700+ price observations
- Full temporal coverage for 365 days
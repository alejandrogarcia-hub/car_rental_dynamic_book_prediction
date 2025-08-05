# Kumo.ai RFM Evaluation for Book or Wait Decision System

## Executive Summary

This document provides a comprehensive evaluation of Kumo.ai's Relational Foundation Model (RFM) for implementing our "Book or Wait" car rental pricing decision system. After extensive research and analysis, I present the findings, feasibility assessment, and implementation recommendations.

## Table of Contents

1. [Kumo.ai RFM Overview](#kumoai-rfm-overview)
2. [Technical Architecture Analysis](#technical-architecture-analysis)
3. [Data Requirements and Compatibility](#data-requirements-and-compatibility)
4. [Predictive Query Language (PQL) Analysis](#predictive-query-language-pql-analysis)
5. [Complexity Assessment](#complexity-assessment)
6. [Required Changes for Book or Wait](#required-changes-for-book-or-wait)
7. [Advantages and Limitations](#advantages-and-limitations)
8. [Cost-Benefit Analysis](#cost-benefit-analysis)
9. [Final Recommendation](#final-recommendation)
10. [Implementation Proposal](#implementation-proposal)

## Kumo.ai RFM Overview

### What is Kumo RFM?

Kumo RFM (Relational Foundation Model) is a groundbreaking AI model designed specifically for making predictions on relational databases without requiring task-specific training. It represents a paradigm shift in how we approach predictive analytics on structured data.

### Key Capabilities

1. **In-Context Learning**: Makes predictions without custom model training by learning from patterns in your existing data
2. **Multi-Table Support**: Naturally handles complex relational structures with multiple interconnected tables
3. **Versatile Predictions**: Supports regression, classification, multi-label classification, and link prediction
4. **SQL-Like Interface**: Uses Predictive Query Language (PQL) that resembles SQL for defining prediction tasks
5. **Instant Results**: Generates predictions in under a second for most queries

### How It Works

1. **Graph Transformation**: Converts relational databases into temporal, heterogeneous graphs
2. **Relational Graph Transformer**: Uses attention mechanisms across:
   - Node types (entities in your tables)
   - Temporal information (time-based patterns)
   - Structural proximity (relationships between entities)
   - Local subgraph patterns
3. **Dynamic Context Generation**: Automatically samples relevant historical examples for in-context learning
4. **Zero-Shot Predictions**: No training required - works immediately on new datasets

## Technical Architecture Analysis

### Core Components

#### 1. LocalTable

```python
import kumoai.experimental.rfm as rfm

# Wraps pandas DataFrame with metadata
table = rfm.LocalTable(
    df=dataframe,
    table_name="rental_prices",
    primary_key="price_id",
    time_column="obs_ts"
)
```

#### 2. LocalGraph

```python
# Creates relational structure
graph = rfm.LocalGraph(tables=[users, bookings, prices])
graph.link(src_table="bookings", fkey="user_id", dst_table="users")
graph.link(src_table="bookings", fkey="price_id", dst_table="prices")
```

#### 3. KumoRFM Model

```python
# Initialize model and make predictions
model = rfm.KumoRFM(graph)
result = model.predict(pql_query)
```

### Data Type System

**Kumo Dtype (Physical Types)**:

- `int`: Integer values
- `float`: Decimal numbers
- `string`: Text data

**Kumo Stype (Semantic Types)**:

- `numerical`: Continuous values (prices, counts)
- `categorical`: Discrete categories (car_class, location)
- `ID`: Entity identifiers (user_id, booking_id)
- `text`: Free-form text (reviews, descriptions)

## Data Requirements and Compatibility

### Current Data Structure Analysis

Our car rental data is well-structured for Kumo RFM:

**Available Tables**:

1. `users` (user_id, segment, home_location_id)
2. `bookings` (booking_id, user_id, supplier_id, prices, timestamps)
3. `rental_prices` (price_id, location_id, supplier_id, car_class, prices, timestamps)
4. `searches` (search_id, user_id, location_id, timestamps)
5. `competitor_prices` (location_id, car_class, prices, timestamps)

**Compatibility Assessment**: ✅ **EXCELLENT**

- Clear primary keys in all tables
- Well-defined foreign key relationships
- Temporal columns present (obs_ts, booking_ts, search_ts)
- Proper entity relationships for graph construction

### Required Data Transformations

1. **Minimal Changes Needed**:

   - Ensure consistent datetime formats
   - Validate foreign key integrity
   - Add explicit primary keys where missing

2. **Graph Structure**:

```text
users <-- bookings --> rental_prices
  |                        |
  +---- searches ----------+
           |
    competitor_prices
```

## Predictive Query Language (PQL) Analysis

### PQL for Book or Wait

The "Book or Wait" decision can be expressed elegantly in PQL:

#### Query 1: Predict if current price is lowest in next 7 days

```sql
PREDICT MIN(rental_prices.current_price, 0, 7, days) > rental_prices.current_price
FOR EACH rental_prices.price_id
WHERE rental_prices.obs_ts = NOW()
```

#### Query 2: Predict price trend

```sql
PREDICT AVG(rental_prices.current_price, 0, 7, days) - rental_prices.current_price
FOR EACH (rental_prices.supplier_id, rental_prices.location_id, rental_prices.car_class)
WHERE rental_prices.days_until_pickup BETWEEN 7 AND 30
```

#### Query 3: Customer-specific booking recommendation

```sql
PREDICT COUNT(bookings.*, 0, 7, days) > 0
FOR EACH users.user_id
WHERE EXISTS(searches.* IN LAST 24 hours)
  AND rental_prices.current_price < competitor_prices.comp_min_price
```

### PQL Advantages for Our Use Case

1. **Time Window Support**: Native handling of temporal predictions (0-7 days ahead)
2. **Aggregation Functions**: SUM, AVG, MIN, MAX perfect for price analysis
3. **Multi-Entity Predictions**: Can predict for users, locations, or price points
4. **Conditional Logic**: WHERE clauses for context filtering

## Complexity Assessment

### Implementation Complexity: **MEDIUM-LOW**

#### Easy Aspects ✅

1. **No Model Training**: Zero training time - immediate predictions
2. **Simple API**: Clean Python SDK with intuitive interface
3. **SQL-Like Syntax**: PQL is familiar to anyone who knows SQL
4. **Automatic Feature Engineering**: RFM handles feature extraction internally

#### Moderate Aspects ⚠️

1. **API Key Required**: Need to register and obtain access
2. **Graph Design**: Requires thoughtful table relationship planning
3. **PQL Learning Curve**: New syntax to master (though SQL-similar)
4. **Limited Documentation**: Being experimental, some features lack examples

#### Challenging Aspects ❌

1. **Black Box Nature**: Limited interpretability compared to XGBoost
2. **Customization Constraints**: Can't modify model architecture
3. **Dependency on External Service**: Requires API calls to Kumo
4. **Cost Uncertainty**: Pricing model not publicly available

## Required Changes for Book or Wait

### 1. Data Pipeline Changes

**Current State**:

```python
# Traditional ML pipeline
features_df = create_price_features(rental_prices_df, competitor_prices_df)
X_train, y_train = prepare_training_data(features_df)
model = XGBClassifier().fit(X_train, y_train)
```

**Kumo RFM Approach**:

```python
# Graph-based approach
graph = create_rental_graph(users_df, bookings_df, prices_df)
model = rfm.KumoRFM(graph)
prediction = model.predict(pql_query)
```

### 2. Feature Engineering Changes

**Eliminated**: Manual feature engineering

- No need for price_change_1d, price_change_7d calculations
- No manual temporal feature extraction
- No competitor price aggregations

**Replaced By**: PQL expressions

- Temporal patterns captured by PQL time windows
- Relationships captured by graph structure
- Aggregations handled by PQL functions

### 3. Model Training Changes

**Eliminated Completely**:

- No train/test split
- No hyperparameter tuning
- No cross-validation
- No model persistence

**Replaced By**:

- Direct predictions via API
- In-context learning at inference time
- Dynamic adaptation to new patterns

### 4. Infrastructure Changes

**New Requirements**:

- Kumo API integration
- Graph construction utilities
- PQL query templates
- Result parsing logic

## Advantages and Limitations

### Advantages for Book or Wait

1. **Zero Training Time** ⭐
   - Immediate deployment without 2-hour training cycles
   - No GPU/compute requirements
   - Instant adaptation to new data patterns

2. **Natural Temporal Handling** ⭐
   - Built-in support for time-based predictions
   - Native "next 7 days" semantics in PQL
   - Automatic temporal pattern recognition

3. **Simplified Pipeline** ⭐
   - No feature engineering code to maintain
   - No model versioning complexity
   - Reduced codebase by ~70%

4. **Multi-Entity Predictions** ⭐
   - Can predict for users, locations, or suppliers
   - Single model handles all entity types
   - Cross-entity pattern learning

5. **Automatic Relationship Learning** ⭐
   - Discovers patterns across table relationships
   - No manual join operations needed
   - Captures complex interactions naturally

### Limitations and Concerns

1. **Interpretability Loss** ⚠️
   - Can't inspect feature importance like XGBoost
   - Black box predictions
   - Harder to explain to stakeholders

2. **External Dependency** ⚠️
   - Requires internet connectivity
   - Subject to API availability
   - Potential latency issues

3. **Cost Uncertainty** ⚠️
   - Pricing not publicly disclosed
   - Potential per-prediction costs
   - Budget impact unknown

4. **Limited Customization** ⚠️
   - Can't modify model behavior
   - Fixed to PQL capabilities
   - No custom loss functions

5. **Experimental Status** ⚠️
   - SDK marked as experimental
   - Potential API changes
   - Limited production references

## Cost-Benefit Analysis

### Benefits

1. **Development Time Savings**:
   - Estimated 60-80% reduction in implementation time
   - No model training/tuning cycles
   - Simplified maintenance

2. **Operational Savings**:
   - No GPU infrastructure needed
   - Reduced compute costs
   - Minimal storage requirements

3. **Flexibility Gains**:
   - Easy to test new prediction tasks
   - Quick experimentation with PQL
   - Rapid iteration on business logic

### Costs

1. **API Costs**: Unknown (requires quote from Kumo)
2. **Migration Effort**: ~2-3 weeks for full transition
3. **Team Training**: ~1 week for PQL proficiency
4. **Risk Mitigation**: Parallel run with existing models

### ROI Estimate

**Break-even**: If API costs < $2,000/month, positive ROI within 3 months due to:

- Reduced development time
- Lower infrastructure costs
- Faster time-to-market for new features

## Final Recommendation

### Recommendation: **CAUTIOUS YES** ✅

**Kumo RFM is feasible and potentially beneficial for the Book or Wait system, with caveats.**

### Rationale

**Pros Outweigh Cons For Our Use Case**:

1. Our relational data structure is perfect for RFM
2. Time-based predictions are a core RFM strength
3. Significant reduction in code complexity
4. Faster experimentation and deployment

**Risk Mitigation Strategy**:

1. Start with pilot implementation
2. Run parallel to existing XGBoost model
3. Compare performance over 1-month period
4. Gradual migration if results are positive

### Conditions for Success

1. **API costs must be reasonable** (< $2,000/month)
2. **Performance must match XGBoost** (ROC-AUC > 0.90)
3. **Latency must be acceptable** (< 100ms per prediction)
4. **Kumo must provide production SLA**

## Implementation Proposal

### Phase 1: Proof of Concept (Week 1-2)

#### 1.1 Environment Setup

```python
# Install Kumo SDK
pip install kumoai

# Configure API access
import kumoai as kumo
kumo.init(api_key=os.environ['KUMO_API_KEY'])
```

#### 1.2 Data Preparation

```python
import kumoai.experimental.rfm as rfm
import pandas as pd

# Load our existing data
users_df = pd.read_csv('data/processed/synthetic_users.csv')
bookings_df = pd.read_csv('data/processed/synthetic_bookings.csv')
prices_df = pd.read_csv('data/processed/synthetic_rental_prices.csv')
searches_df = pd.read_csv('data/processed/synthetic_searches.csv')
competitor_df = pd.read_csv('data/processed/synthetic_competitor_prices.csv')
```

#### 1.3 Graph Construction

```python
# Create LocalTables with metadata
users = rfm.LocalTable(
    users_df, 
    name="users",
    primary_key="user_id"
).infer_metadata()

bookings = rfm.LocalTable(
    bookings_df,
    name="bookings", 
    primary_key="booking_id",
    time_column="booking_ts"
).infer_metadata()

prices = rfm.LocalTable(
    prices_df,
    name="prices",
    primary_key="price_id",
    time_column="obs_ts"
).infer_metadata()

# Create graph and define relationships
graph = rfm.LocalGraph(tables=[users, bookings, prices, searches, competitor])

# Link tables
graph.link(src_table="bookings", fkey="user_id", dst_table="users")
graph.link(src_table="bookings", fkey="search_id", dst_table="searches")
graph.link(src_table="searches", fkey="user_id", dst_table="users")
```

#### 1.4 Book or Wait Predictions

```python
# Initialize model
model = rfm.KumoRFM(graph)

# Query 1: Will price increase in next 7 days?
book_or_wait_query = """
PREDICT 
    CASE 
        WHEN MIN(prices.current_price, 0, 7, days) > prices.current_price THEN 1
        ELSE 0
    END AS should_book_now
FOR EACH prices.price_id
WHERE prices.days_until_pickup > 7
"""

# Make predictions
predictions = model.predict(book_or_wait_query)
```

### Phase 2: Performance Evaluation (Week 3)

#### 2.1 Comparison Framework

```python
class KumoRFMEvaluator:
    def __init__(self, kumo_model, xgboost_model):
        self.kumo = kumo_model
        self.xgb = xgboost_model
        
    def compare_predictions(self, test_data):
        # Get Kumo predictions
        kumo_preds = self.kumo.predict(self.build_pql_query(test_data))
        
        # Get XGBoost predictions
        xgb_features = create_price_features(test_data)
        xgb_preds = self.xgb.predict_proba(xgb_features)[:, 1]
        
        # Calculate metrics
        kumo_auc = roc_auc_score(test_data['target'], kumo_preds)
        xgb_auc = roc_auc_score(test_data['target'], xgb_preds)
        
        return {
            'kumo_auc': kumo_auc,
            'xgb_auc': xgb_auc,
            'difference': kumo_auc - xgb_auc
        }
```

#### 2.2 Latency Testing

```python
import time

def measure_latency(model, queries, n_iterations=100):
    latencies = []
    
    for _ in range(n_iterations):
        query = random.choice(queries)
        start = time.time()
        _ = model.predict(query)
        latencies.append(time.time() - start)
    
    return {
        'mean_latency': np.mean(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'max_latency': np.max(latencies)
    }
```

### Phase 3: Production Integration (Week 4)

#### 3.1 API Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class BookOrWaitRequest(BaseModel):
    user_id: int
    supplier_id: int
    location_id: int
    car_class: str
    current_price: float
    days_until_pickup: int

@app.post("/predict/book-or-wait")
async def predict_book_or_wait(request: BookOrWaitRequest):
    try:
        # Build PQL query for specific request
        query = f"""
        PREDICT 
            CASE 
                WHEN AVG(prices.current_price, 0, 7, days) > {request.current_price} 
                THEN 1 
                ELSE 0 
            END AS should_book
        FOR prices.supplier_id = {request.supplier_id}
            AND prices.location_id = {request.location_id}
            AND prices.car_class = '{request.car_class}'
        WHERE prices.days_until_pickup >= {request.days_until_pickup}
        """
        
        # Get prediction
        result = kumo_model.predict(query)
        
        return {
            "should_book_now": bool(result['should_book'].iloc[0]),
            "confidence": float(result.get('confidence', 0.0)),
            "model": "kumo_rfm"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3.2 A/B Testing Framework

```python
class BookOrWaitABTest:
    def __init__(self, kumo_model, xgb_model, traffic_split=0.5):
        self.kumo = kumo_model
        self.xgb = xgb_model
        self.traffic_split = traffic_split
        
    def get_recommendation(self, request):
        # Randomly assign to model
        use_kumo = random.random() < self.traffic_split
        
        if use_kumo:
            prediction = self.kumo_predict(request)
            model_used = "kumo_rfm"
        else:
            prediction = self.xgb_predict(request)
            model_used = "xgboost"
        
        # Log for analysis
        self.log_prediction(request, prediction, model_used)
        
        return prediction, model_used
```

### Phase 4: Monitoring and Optimization (Ongoing)

#### 4.1 Performance Monitoring

```python
class KumoPerformanceMonitor:
    def track_metrics(self):
        return {
            'daily_predictions': self.count_daily_predictions(),
            'average_latency': self.calculate_avg_latency(),
            'error_rate': self.calculate_error_rate(),
            'api_availability': self.check_api_health(),
            'cost_per_prediction': self.calculate_unit_cost()
        }
```

#### 4.2 Cost Optimization

```python
class PredictionCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get_or_predict(self, query, model):
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                return result
        
        # Make fresh prediction
        result = model.predict(query)
        self.cache[cache_key] = (result, time.time())
        return result
```

### Success Metrics

1. **Performance Metrics**:
   - ROC-AUC ≥ 0.90 (matching XGBoost)
   - Precision on "Wait" class ≥ 0.75
   - Recall on "Book Now" class ≥ 0.90

2. **Operational Metrics**:
   - API latency < 100ms (p95)
   - API availability > 99.9%
   - Cost per prediction < $0.001

3. **Business Metrics**:
   - Customer savings rate maintained
   - A/B test shows no degradation
   - Reduced development time by 60%

### Risk Mitigation

1. **Fallback Strategy**: Keep XGBoost model as backup
2. **Gradual Rollout**: Start with 10% traffic, increase gradually
3. **Data Quality Checks**: Validate graph integrity daily
4. **Cost Controls**: Set spending limits and alerts

## Conclusion

Kumo.ai RFM presents a compelling alternative to our current XGBoost implementation for the Book or Wait system. The technology is well-suited to our use case, offering significant advantages in development speed, maintenance, and flexibility. However, careful evaluation of costs, performance, and reliability is essential before full commitment.

I recommend proceeding with a proof-of-concept implementation while maintaining our existing model as a safety net. This approach allows us to validate the technology's promises while minimizing risk to our production system.

# Data Sourcing Module

This module handles data acquisition and loading for the Credit Risk Analytics pipeline.

## Overview

The data sourcing module is responsible for:
- Loading raw credit data from various sources
- Data validation and integrity checks
- Initial data preprocessing
- Data versioning and tracking

## Files

- `data_loader.py` - Main data loading utilities
- `data_sources.md` - Documentation of data sources
- `validators.py` - Data validation functions

## Usage

```python
from src.data_sourcing.data_loader import load_credit_data

# Load raw credit data
df = load_credit_data(source='local', path='data/raw/credit_data.csv')
```

## Data Sources

Document your data sources in `data_sources.md`.

## TODO

- [ ] Implement database connectors
- [ ] Add API data fetching
- [ ] Set up data versioning with DVC

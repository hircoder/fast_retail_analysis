# Fast Retailing Data Analysis Script
Quick analysis script to analyze time-series data using dummy consumer purchase data.

## What it does
- Loads and validates transaction data from 2018-2020
- Cleans sales data (removes nulls and outliers)
- Extracts Fast Retailing store data (UNIQLO, GU, etc.)
- Aggregates quarterly metrics
- Compares against benchmark disclosure data
- Calculates correlation and error metrics
- Optional: Does normalized user-matched analysis

## Requirements
- Python 3.7+
- Required packages:
 - pandas
 - numpy 
 - scipy
 - matplotlib
 - sklearn

## Input Files
- `raw_data.tsv.gz`: Transaction data
- `disclosure.csv`: Benchmark disclosure data 

## Outputs
Files saved to `output/` directory:
- `brand_store_map.tsv`: Brand-store mapping
- `aggregation.csv`: Quarterly aggregated metrics
- `evaluation_data.csv`: YoY comparison results
- `evaluation.json`: Statistical metrics
- `yoy_comparison.png`: Visualization
- Optional normalized outputs with `_v2` suffix

## Usage
```bash
python fast_retail_analysis.py

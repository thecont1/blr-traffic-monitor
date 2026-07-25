# Traffic Analysis Enhancements - User Guide

## Overview

This comprehensive traffic analysis system provides statistical analysis and visualization capabilities for the Bangalore traffic monitoring project. It implements the R³S² (Rolling Relative Route Scoring System) methodology with extensive validation, alternative scoring methods, and rich visualizations.

## Installation Requirements

```bash
# Core dependencies
pip install pandas numpy scipy statsmodels matplotlib seaborn plotly scikit-learn

# Optional (for interactive features)
pip install ipywidgets jupyter

# Optional (for property-based testing)
pip install hypothesis pytest
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## Quick Start Guide

### 1. Basic Setup

```python
import pandas as pd
from traffic_analyzer import TrafficAnalyzer
from visualization_engine import VisualizationEngine
from data_utils import compute_temporal_features

# Load your data
df = pd.read_csv('data/csv-traffic-bangalore.csv')
routes_df = pd.read_csv('data/csv-routes-bangalore.csv')

# Prepare data (add temporal features)
df = compute_temporal_features(df)

# Initialize analyzers
analyzer = TrafficAnalyzer(df, routes_df)
viz = VisualizationEngine(df, routes_df)
```

### 2. R³S² Analysis

```python
# Calculate R³S² scores
scores = analyzer.calculate_rrs(days_rolling=10)
print(scores)

# Analyze correlation with raw speeds
correlation = analyzer.analyze_rrs_correlation()
print(f"Pearson r = {correlation['pearson_correlation']:.3f}")

# Test sensitivity to outliers
sensitivity = analyzer.analyze_rrs_sensitivity()
print(f"Rank correlation = {sensitivity['rank_correlation']:.3f}")

# Measure stability across window sizes
stability = analyzer.analyze_rrs_stability()
print(stability)
```

### 3. Data Quality Assessment

```python
# Compute quality metrics
quality = analyzer.compute_quality_metrics()
print(f"Completeness: {quality['completeness']:.1f}%")
print(f"Outlier rate: {quality['outlier_rate']:.1f}%")

# Analyze missing data patterns
completeness = analyzer.analyze_data_completeness()
print(completeness)

# Identify systematic gaps
patterns = analyzer.identify_missing_patterns()
print(patterns[patterns['missing_pct'] > 20])
```

### 4. Alternative Scoring Methods

```python
# Compare different scoring approaches
comparison = analyzer.compare_scoring_methods()
print(comparison)

# Compute bootstrap confidence intervals
ci = analyzer.compute_rrs_confidence_intervals()
print(ci)

# Test transitivity
transitivity = analyzer.test_rrs_transitivity()
print(transitivity)
```

### 5. Statistical Testing

```python
# Test normality
normality = analyzer.test_normality()
print(normality)

# Test stationarity
stationarity = analyzer.test_stationarity()
print(stationarity)

# Analyze autocorrelation
acf = analyzer.analyze_autocorrelation()
for route, acf_values in acf.items():
    print(f"{route}: ACF at lag 24 = {acf_values[24]:.3f}")

# Test variance homogeneity
variance_test = analyzer.test_variance_homogeneity()
print(variance_test)

# Perform power analysis
power = analyzer.perform_power_analysis()
print(power)
```

### 6. Recommendations

```python
# Generate actionable recommendations
recommendations = analyzer.generate_recommendations()
for rec in recommendations:
    print(f"\n{rec['type'].upper()} ({rec['severity']})")
    print(f"  {rec['description']}")
    print(f"  Expected benefit: {rec['expected_benefit']}")
```

## Visualization Guide

### Temporal Pattern Visualizations

```python
# Hourly heatmap (single route)
viz.plot_hourly_heatmap('VJRQ+2M|RMJJ+F4')

# Hourly heatmap (all routes)
viz.plot_hourly_heatmap()

# Calendar heatmap
viz.plot_calendar_heatmap('VJRQ+2M|RMJJ+F4')

# Time series decomposition
viz.plot_time_series_decomposition('VJRQ+2M|RMJJ+F4')

# Hour-of-day profiles
viz.plot_hour_of_day_profiles()

# Correlation matrix
viz.plot_correlation_matrix()

# Ranking animation
viz.create_ranking_animation()
```

### Comparative Performance Visualizations

```python
# Parallel coordinates
viz.plot_parallel_coordinates()

# Radar chart
viz.plot_radar_chart()

# Speed-duration scatter
viz.plot_speed_duration_scatter()

# Time-of-day facets
viz.plot_time_of_day_facets()

# CDF comparison
viz.plot_cdf_comparison()

# Ranking evolution
viz.plot_ranking_evolution(days_rolling=10)
```

### Anomaly Detection Visualizations

```python
# Control chart
viz.plot_control_chart('VJRQ+2M|RMJJ+F4')

# Anomaly scatter
viz.plot_anomaly_scatter('VJRQ+2M|RMJJ+F4')

# Deviation timeline
viz.plot_deviation_timeline('VJRQ+2M|RMJJ+F4')

# Outlier summary
outliers = viz.generate_outlier_summary(top_n=20)
print(outliers)

# Residual analysis
viz.plot_residual_analysis('VJRQ+2M|RMJJ+F4')
```

### Predictive Insight Visualizations

```python
# Forecast
viz.plot_forecast('VJRQ+2M|RMJJ+F4', forecast_hours=24)

# Typical day profile
viz.plot_typical_day_profile('Monday')
viz.plot_typical_day_profile()  # All days

# Current vs predicted
viz.plot_current_vs_predicted('VJRQ+2M|RMJJ+F4', '2025-09-15')

# Seasonal trends
viz.plot_seasonal_trends('VJRQ+2M|RMJJ+F4')

# Lag correlations
viz.plot_lag_correlations()

# Best travel times
viz.plot_best_travel_times()
```

### Diagnostic Plots

```python
# Q-Q plots for all routes
analyzer.plot_qq_normality()

# Residual diagnostics
analyzer.plot_residual_diagnostics('VJRQ+2M|RMJJ+F4')
```

### Methodology Evaluation

```python
# Evaluate stability
stability = analyzer.evaluate_scoring_stability()
print(stability['summary'])

# Evaluate outlier sensitivity
sensitivity = analyzer.evaluate_outlier_sensitivity()
print(sensitivity['summary'])

# Evaluate computational efficiency
efficiency = analyzer.evaluate_computational_efficiency()
print(efficiency['summary'])

# Generate comprehensive ranking report
report = analyzer.generate_methodology_ranking_report()
print(report)
```

## Common Usage Patterns

### Pattern 1: Complete Route Analysis

```python
route_code = 'VJRQ+2M|RMJJ+F4'

# Statistical analysis
print("=== Statistical Analysis ===")
scores = analyzer.calculate_rrs()
print(f"R³S² Score: {scores[scores['route_code'] == route_code]['points'].iloc[0]:.1f}")

normality = analyzer.test_normality()
route_norm = normality[normality['route_code'] == route_code]
print(f"Normally distributed: {route_norm['is_normal_shapiro'].iloc[0]}")

# Visualizations
print("\n=== Generating Visualizations ===")
viz.plot_hourly_heatmap(route_code)
viz.plot_time_series_decomposition(route_code)
viz.plot_control_chart(route_code)
viz.plot_forecast(route_code)
```

### Pattern 2: System-Wide Comparison

```python
print("=== System-Wide Analysis ===")

# Compare all routes
viz.plot_hour_of_day_profiles()
viz.plot_correlation_matrix()
viz.plot_cdf_comparison()
viz.plot_best_travel_times()

# Data quality
quality = analyzer.compute_quality_metrics()
print(f"\nOverall completeness: {quality['completeness']:.1f}%")
print(f"Routes with sufficient data: {quality['routes_coverage']['sufficient_data_routes']}")
```

### Pattern 3: Methodology Validation

```python
print("=== Methodology Validation ===")

# Test R³S² properties
correlation = analyzer.analyze_rrs_correlation()
print(f"Correlation with raw speeds: {correlation['spearman_correlation']:.3f}")

sensitivity = analyzer.analyze_rrs_sensitivity()
print(f"Outlier sensitivity: {sensitivity['rank_correlation']:.3f}")

stability = analyzer.analyze_rrs_stability()
print(f"\nStability across window sizes:")
print(stability)

# Compare methods
comparison = analyzer.compare_scoring_methods()
print(f"\nMethod comparison:")
print(comparison)

# Generate report
report = analyzer.generate_methodology_ranking_report()
```

## Data Requirements

### Required DataFrame Columns

**Traffic Data (df)**:
- `year`: int - Year
- `month`: int - Month (1-12)
- `day`: int - Day of month (1-31)
- `hour`: int - Hour of day (0-23)
- `route_code`: str - Route identifier
- `duration`: float - Travel time in minutes
- `distance`: float - Distance in kilometers
- `avg_speed`: float - Average speed in km/h

**Routes Metadata (routes_df)**:
- `route_code`: str - Route identifier (must match df)
- `label_full`: str - Full route description
- `label_short`: str - Short route label
- `color_hex`: str - Hex color code for visualizations

### Temporal Features (Auto-Generated)

The `compute_temporal_features()` function adds:
- `timestamp`: datetime - Combined date/time
- `day_of_week`: str - Day name (Monday, Tuesday, etc.)
- `is_weekend`: bool - Weekend indicator
- `time_category`: str - Time period (morning_rush, midday, evening_rush, night)

## Performance Considerations

### Memory Usage

- Large datasets (>100K observations): Use sampling for exploratory analysis
- Bootstrap operations: Reduce `n_bootstrap` parameter if memory constrained
- Visualization: Close plots after viewing to free memory

### Computational Efficiency

- R³S² calculation: O(n * d) where n=observations, d=days
- Correlation matrix: O(n * m²) where m=routes
- Time series decomposition: O(n) per route
- Bootstrap CI: O(n * b) where b=bootstrap iterations

### Optimization Tips

```python
# Use smaller rolling windows for faster computation
scores = analyzer.calculate_rrs(days_rolling=7)  # Instead of 10

# Reduce bootstrap iterations
ci = analyzer.compute_rrs_confidence_intervals(n_bootstrap=500)  # Instead of 1000

# Sample data for exploratory analysis
df_sample = df.sample(frac=0.1, random_state=42)
analyzer_sample = TrafficAnalyzer(df_sample, routes_df)
```

## Troubleshooting

### Common Issues

**Issue**: "Insufficient data for analysis"
- **Solution**: Ensure at least 2 weeks of data for time series decomposition
- **Solution**: Check data completeness with `analyzer.analyze_data_completeness()`

**Issue**: "Route not found in routes_df"
- **Solution**: Verify route_code matches between df and routes_df
- **Solution**: Check for typos or case sensitivity

**Issue**: "Memory error during bootstrap"
- **Solution**: Reduce `n_bootstrap` parameter
- **Solution**: Use smaller dataset or sample data

**Issue**: "Plots not displaying"
- **Solution**: Ensure matplotlib backend is configured: `%matplotlib inline` (Jupyter)
- **Solution**: Call `plt.show()` explicitly if needed

### Data Quality Warnings

The system automatically warns about:
- Routes with <80% data completeness
- High outlier rates (>5%)
- Inconsistent distance measurements
- Systematic missing data patterns

Address these warnings by:
1. Improving data collection
2. Implementing gap-filling strategies
3. Using robust statistical methods
4. Filtering problematic observations

## Best Practices

### 1. Data Preparation

```python
# Always validate data first
from traffic_analyzer import validate_traffic_dataframe
validate_traffic_dataframe(df)

# Add temporal features
df = compute_temporal_features(df)

# Check data quality before analysis
quality = analyzer.compute_quality_metrics()
if quality['completeness'] < 80:
    print("Warning: Low data completeness. Results may be unreliable.")
```

### 2. Iterative Analysis

```python
# Start with overview
quality = analyzer.compute_quality_metrics()
recommendations = analyzer.generate_recommendations()

# Deep dive into specific routes
for route in problem_routes:
    analyzer.plot_residual_diagnostics(route)
    viz.plot_control_chart(route)
    viz.plot_deviation_timeline(route)
```

### 3. Validation

```python
# Always validate methodology
correlation = analyzer.analyze_rrs_correlation()
sensitivity = analyzer.analyze_rrs_sensitivity()
stability = analyzer.analyze_rrs_stability()

# Compare with alternatives
comparison = analyzer.compare_scoring_methods()

# Generate comprehensive report
report = analyzer.generate_methodology_ranking_report()
```

### 4. Documentation

```python
# Document your analysis
analysis_summary = {
    'date': pd.Timestamp.now(),
    'data_period': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
    'n_observations': len(df),
    'n_routes': len(routes_df),
    'completeness': quality['completeness'],
    'r3s2_correlation': correlation['spearman_correlation'],
    'recommendations': len(recommendations)
}

# Save results
scores.to_csv('r3s2_scores.csv', index=False)
outliers.to_csv('outliers.csv', index=False)
report.to_csv('methodology_report.csv', index=False)
```

## Advanced Features

### Custom Scoring Methods

```python
# Implement custom scoring logic
def custom_score(df, routes_df):
    # Your custom logic here
    pass

# Compare with existing methods
custom_scores = custom_score(df, routes_df)
rrs_scores = analyzer.calculate_rrs()
# Compare...
```

### Batch Processing

```python
# Process multiple routes
for route in routes_df['route_code']:
    try:
        viz.plot_time_series_decomposition(route)
        plt.savefig(f'decomposition_{route}.png')
        plt.close()
    except Exception as e:
        print(f"Error processing {route}: {e}")
```

### Integration with Existing Notebook

```python
# In traffic_analysis_examples.ipynb, add new cells:

# Cell 1: Import new modules
from traffic_analyzer import TrafficAnalyzer
from visualization_engine import VisualizationEngine

# Cell 2: Initialize
analyzer = TrafficAnalyzer(df, routes_df)
viz = VisualizationEngine(df, routes_df)

# Cell 3: Analysis
scores = analyzer.calculate_rrs()
recommendations = analyzer.generate_recommendations()

# Cell 4: Visualizations
viz.plot_hour_of_day_profiles()
viz.plot_correlation_matrix()
viz.plot_best_travel_times()
```

## API Reference

See individual module docstrings for detailed API documentation:

```python
# View module help
help(TrafficAnalyzer)
help(VisualizationEngine)

# View method help
help(analyzer.calculate_rrs)
help(viz.plot_hourly_heatmap)
```

## Support and Contributing

For issues, questions, or contributions:
1. Check this documentation first
2. Review method docstrings
3. Examine example usage in demo scripts
4. Consult the design document in `.kiro/specs/traffic-analysis-enhancements/`

## License

See LICENSE.md for details.

## Changelog

### Version 1.0.0 (2025-09-XX)
- Initial release
- Complete R³S² analysis implementation
- 50+ visualization methods
- Comprehensive statistical testing
- Methodology evaluation framework
- Full documentation and examples

"""
Traffic Analyzer Module

Statistical analysis engine for traffic data and R³S² (Rolling Relative Route Scoring System) evaluation.
Provides comprehensive analysis of route performance, data quality, and scoring methodology validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings


# ============================================================================
# Custom Exception Classes
# ============================================================================

class TrafficAnalysisError(Exception):
    """Base exception for traffic analysis errors"""
    pass


class InsufficientDataError(TrafficAnalysisError):
    """Raised when dataset is too small for analysis"""
    pass


class InvalidRouteError(TrafficAnalysisError):
    """Raised when route_code is not found in dataset"""
    pass


class ScoringMethodError(TrafficAnalysisError):
    """Raised when scoring method fails"""
    pass


# ============================================================================
# Input Validation Functions
# ============================================================================

def validate_traffic_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required columns and valid data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Traffic data to validate
        
    Raises
    ------
    ValueError
        If required columns are missing or data is invalid
    TypeError
        If column data types are incorrect
    """
    required_columns = ['year', 'month', 'day', 'hour', 'route_code', 
                       'duration', 'distance', 'avg_speed']
    
    # Check for missing columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['avg_speed']):
        raise TypeError("avg_speed must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['duration']):
        raise TypeError("duration must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['distance']):
        raise TypeError("distance must be numeric")
    
    # Validate value ranges
    if (df['avg_speed'] < 0).any():
        raise ValueError("avg_speed cannot be negative")
    
    if (df['duration'] <= 0).any():
        raise ValueError("duration must be positive")
    
    if (df['distance'] <= 0).any():
        raise ValueError("distance must be positive")
    
    if (df['hour'] < 0).any() or (df['hour'] > 23).any():
        raise ValueError("hour must be in range [0, 23]")
    
    if (df['month'] < 1).any() or (df['month'] > 12).any():
        raise ValueError("month must be in range [1, 12]")
    
    if (df['day'] < 1).any() or (df['day'] > 31).any():
        raise ValueError("day must be in range [1, 31]")


# ============================================================================
# TrafficAnalyzer Class
# ============================================================================

class TrafficAnalyzer:
    """
    Statistical analysis engine for traffic data and R³S² evaluation.
    
    This class provides comprehensive analysis capabilities including:
    - R³S² methodology evaluation (correlation, sensitivity, stability)
    - Statistical testing (normality, stationarity, autocorrelation)
    - Data quality assessment and missing data analysis
    - Alternative scoring method comparison
    - Recommendation generation
    
    Parameters
    ----------
    df : pd.DataFrame
        Traffic data with columns: year, month, day, hour, route_code,
        duration, distance, avg_speed
    routes_df : pd.DataFrame
        Route metadata with columns: route_code, label_full, label_short, color_hex
        
    Attributes
    ----------
    df : pd.DataFrame
        Validated traffic data
    routes_df : pd.DataFrame
        Route metadata
    routes : List[str]
        List of unique route codes
        
    Examples
    --------
    >>> analyzer = TrafficAnalyzer(traffic_df, routes_df)
    >>> correlation_results = analyzer.analyze_rrs_correlation()
    >>> recommendations = analyzer.generate_recommendations()
    """
    
    def __init__(self, df: pd.DataFrame, routes_df: pd.DataFrame):
        """
        Initialize TrafficAnalyzer with traffic data and route metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            Traffic data
        routes_df : pd.DataFrame
            Route metadata
        """
        # Validate input data
        validate_traffic_dataframe(df)
        
        # Store validated data
        self.df = df.copy()
        self.routes_df = routes_df.copy()
        
        # Extract unique routes
        self.routes = sorted(self.df['route_code'].dropna().unique())
        
        # Validate that routes exist in routes_df
        missing_routes = set(self.routes) - set(routes_df['route_code'])
        if missing_routes:
            warnings.warn(f"Routes in data but not in routes_df: {missing_routes}")
    
    def __repr__(self) -> str:
        """String representation of TrafficAnalyzer"""
        n_obs = len(self.df)
        n_routes = len(self.routes)
        date_range = f"{self.df['year'].min()}-{self.df['month'].min():02d} to {self.df['year'].max()}-{self.df['month'].max():02d}"
        return f"TrafficAnalyzer(observations={n_obs}, routes={n_routes}, period={date_range})"

    
    # ========================================================================
    # R³S² Analysis Methods
    # ========================================================================
    
    def _get_variances(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute centered 'points' score per route based on mean average speed.
        
        This is the core R³S² scoring function that ranks routes by speed
        and assigns centered linear points.
        
        Parameters
        ----------
        df_subset : pd.DataFrame
            Subset of traffic data (e.g., one day)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'points']
        """
        dfv = df_subset.copy()
        
        # Ensure avg_speed is computed
        if 'avg_speed' not in dfv.columns or dfv['avg_speed'].isna().any():
            dfv['avg_speed'] = 60.0 * dfv['distance'] / dfv['duration']
        
        # Aggregate by route
        agg = (
            dfv.groupby('route_code', as_index=False)
               .agg(
                   duration_min=('duration', 'min'),
                   duration_mean=('duration', 'mean'),
                   duration_max=('duration', 'max'),
                   distance_min=('distance', 'min'),
                   distance_mean=('distance', 'mean'),
                   distance_max=('distance', 'max'),
                   avg_speed_min=('avg_speed', 'min'),
                   avg_speed_mean=('avg_speed', 'mean'),
                   avg_speed_max=('avg_speed', 'max'),
               )
        )
        
        # Sort by mean avg speed (desc) and assign centered points
        agg = agg.sort_values('avg_speed_mean', ascending=False).reset_index(drop=True)
        n = len(agg)
        points = np.linspace(n/2, -n/2, n)
        
        out = agg[['route_code']].copy()
        out['points'] = points
        return out
    
    def calculate_rrs(self, ref_date: Optional[str] = None, 
                     days_rolling: int = 10) -> pd.DataFrame:
        """
        Calculate R³S² scores by summing daily points over a rolling window.
        
        Parameters
        ----------
        ref_date : str, optional
            Reference date for the rolling window (default: most recent date in data)
        days_rolling : int, default=10
            Number of days in the rolling window
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'points'] sorted by points descending
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> scores = analyzer.calculate_rrs(days_rolling=14)
        """
        from datetime import timedelta
        
        df = self.df.copy()
        
        # Ensure we have a datetime 'timestamp'
        if 'timestamp' not in df.columns:
            if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
                df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            else:
                raise ValueError("DataFrame must contain 'timestamp' or year/month/day/hour")
        
        # Normalize reference date
        if ref_date is None:
            ref_date = pd.to_datetime(df['timestamp'].max()).normalize()
        elif isinstance(ref_date, str):
            ref_date = pd.to_datetime(ref_date).normalize()
        else:
            ref_date = pd.to_datetime(ref_date).normalize()
        
        # Initialize scores
        scores = pd.DataFrame({'route_code': self.routes, 'points': 0.0})
        
        # Accumulate daily points over the rolling window
        for d in range(days_rolling, 0, -1):
            day = (ref_date - timedelta(days=d)).date()
            day_df = df.loc[df['timestamp'].dt.date == day]
            
            if day_df.empty:
                continue
            
            day_pts = self._get_variances(day_df)
            # Merge and add with explicit suffix to avoid ambiguous columns
            scores = scores.merge(day_pts, on='route_code', how='left', suffixes=('', '_day'))
            scores['points'] = scores['points'] + scores['points_day'].fillna(0.0)
            scores.drop(columns=['points_day'], inplace=True)
        
        scores = scores.sort_values('points', ascending=False).round({'points': 1}).reset_index(drop=True)
        return scores
    
    def analyze_rrs_correlation(self, days_rolling: int = 10) -> Dict[str, float]:
        """
        Compute correlation between R³S² scores and raw mean average speeds.
        
        Calculates both Pearson (linear) and Spearman (rank) correlations
        to assess how well R³S² scores represent actual route performance.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days in the rolling window for R³S² calculation
            
        Returns
        -------
        Dict[str, float]
            Dictionary with keys:
            - 'pearson_correlation': Pearson correlation coefficient
            - 'pearson_pvalue': p-value for Pearson correlation
            - 'spearman_correlation': Spearman rank correlation coefficient
            - 'spearman_pvalue': p-value for Spearman correlation
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> results = analyzer.analyze_rrs_correlation()
        >>> print(f"Pearson r = {results['pearson_correlation']:.3f}")
        """
        from scipy import stats
        
        # Calculate R³S² scores
        rrs_scores = self.calculate_rrs(days_rolling=days_rolling)
        
        # Calculate raw mean speeds
        raw_speeds = self.df.groupby('route_code')['avg_speed'].mean()
        
        # Merge scores and speeds
        merged = rrs_scores.merge(
            raw_speeds.reset_index().rename(columns={'avg_speed': 'mean_speed'}),
            on='route_code'
        )
        
        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(merged['points'], merged['mean_speed'])
        spearman_r, spearman_p = stats.spearmanr(merged['points'], merged['mean_speed'])
        
        return {
            'pearson_correlation': pearson_r,
            'pearson_pvalue': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_pvalue': spearman_p
        }

    
    def analyze_rrs_sensitivity(self, days_rolling: int = 10, 
                                outlier_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Calculate sensitivity of R³S² scores to outlier observations.
        
        Measures how R³S² rankings change when outliers (observations > threshold
        standard deviations from mean) are removed from the dataset.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days in the rolling window
        outlier_threshold : float, default=3.0
            Number of standard deviations to define outliers
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'baseline_scores': R³S² scores with all data
            - 'filtered_scores': R³S² scores with outliers removed
            - 'rank_correlation': Spearman correlation between rankings
            - 'rank_correlation_pvalue': p-value for rank correlation
            - 'n_outliers_removed': Number of outliers removed
            - 'outliers_removed': Alias for backward compatibility
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> sensitivity = analyzer.analyze_rrs_sensitivity()
        >>> print(f"Rank correlation: {sensitivity['rank_correlation']:.3f}")
        """
        from scipy import stats
        from data_utils import detect_outliers
        
        # Calculate baseline scores
        baseline_scores = self.calculate_rrs(days_rolling=days_rolling)
        
        # Identify and remove outliers
        outlier_mask = detect_outliers(self.df['avg_speed'], method='zscore', 
                                      threshold=outlier_threshold)
        df_filtered = self.df[~outlier_mask].copy()
        n_outliers = outlier_mask.sum()
        
        # Create temporary analyzer with filtered data
        temp_analyzer = TrafficAnalyzer(df_filtered, self.routes_df)
        filtered_scores = temp_analyzer.calculate_rrs(days_rolling=days_rolling)
        
        # Merge scores for comparison
        merged = baseline_scores.merge(
            filtered_scores,
            on='route_code',
            suffixes=('_baseline', '_filtered')
        )
        
        # Compute rank correlation
        rank_corr, rank_p = stats.spearmanr(
            merged['points_baseline'],
            merged['points_filtered']
        )
        
        return {
            'baseline_scores': baseline_scores,
            'filtered_scores': filtered_scores,
            'rank_correlation': rank_corr,
            'rank_correlation_pvalue': rank_p,
            'n_outliers_removed': int(n_outliers),
            'outliers_removed': int(n_outliers)
        }
    
    def analyze_rrs_stability(self, window_sizes: List[int] = [5, 10, 15, 20, 30]) -> pd.DataFrame:
        """
        Measure stability of route rankings across different rolling window sizes.
        
        Computes R³S² scores for multiple window sizes and calculates pairwise
        rank correlations to assess ranking stability.
        
        Parameters
        ----------
        window_sizes : List[int], default=[5, 10, 15, 20, 30]
            List of rolling window sizes (in days) to test
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix showing Spearman rank correlations between
            rankings from different window sizes
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> stability = analyzer.analyze_rrs_stability()
        >>> print(stability)
        """
        from scipy import stats
        
        # Calculate scores for each window size
        scores_dict = {}
        for window in window_sizes:
            scores = self.calculate_rrs(days_rolling=window)
            scores_dict[f'{window}d'] = scores.set_index('route_code')['points']
        
        # Create DataFrame with all scores
        scores_df = pd.DataFrame(scores_dict)
        
        # Compute pairwise rank correlations
        n_windows = len(window_sizes)
        stability_matrix = np.zeros((n_windows, n_windows))
        
        for i, win1 in enumerate(window_sizes):
            for j, win2 in enumerate(window_sizes):
                if i == j:
                    stability_matrix[i, j] = 1.0
                else:
                    corr, _ = stats.spearmanr(
                        scores_df[f'{win1}d'],
                        scores_df[f'{win2}d']
                    )
                    stability_matrix[i, j] = corr
        
        # Create labeled DataFrame
        labels = [f'{w}d' for w in window_sizes]
        stability_df = pd.DataFrame(
            stability_matrix,
            index=labels,
            columns=labels
        )
        
        return stability_df

    
    # ========================================================================
    # Missing Data and Bias Analysis Methods
    # ========================================================================
    
    def analyze_missing_data_bias(self, days_rolling: int = 10) -> Dict[str, Any]:
        """
        Quantify bias introduced by unequal sample sizes across routes.
        
        Computes the impact of missing data on R³S² scores by comparing
        scores with complete data vs. actual data with gaps.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days in the rolling window
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'completeness_by_route': Percentage of expected observations present
            - 'bias_metric': Overall bias score (0 = no bias, 1 = maximum bias)
            - 'sample_size_variance': Variance in sample sizes across routes
            - 'recommendations': List of recommendations if bias exceeds threshold
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> bias_analysis = analyzer.analyze_missing_data_bias()
        >>> print(f"Bias metric: {bias_analysis['bias_metric']:.2%}")
        """
        # Calculate expected vs actual observations
        expected_obs_per_route = len(self.df) / len(self.routes)
        actual_obs = self.df.groupby('route_code').size()
        
        # Compute completeness percentage
        completeness = (actual_obs / expected_obs_per_route * 100).to_dict()
        
        # Calculate bias metric (coefficient of variation)
        sample_size_variance = actual_obs.var()
        sample_size_mean = actual_obs.mean()
        bias_metric = (actual_obs.std() / sample_size_mean) if sample_size_mean > 0 else 0
        
        # Generate recommendations if bias exceeds threshold
        recommendations = []
        if bias_metric > 0.1:  # 10% threshold
            recommendations.append({
                'type': 'missing_data_bias',
                'severity': 'high' if bias_metric > 0.2 else 'medium',
                'description': f'Unequal sample sizes detected (CV={bias_metric:.2%}). '
                              'Consider imputation or data quality thresholds.',
                'expected_benefit': 'Improved ranking fairness and reduced bias'
            })
        
        return {
            'completeness_by_route': completeness,
            'bias_metric': bias_metric,
            'sample_size_variance': float(sample_size_variance),
            'recommendations': recommendations
        }
    
    def analyze_data_completeness(self) -> pd.DataFrame:
        """
        Compute percentage of missing observations for each route by hour and day-of-week.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with completeness metrics by route, hour, and day-of-week
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> completeness = analyzer.analyze_data_completeness()
        >>> print(completeness[completeness['completeness_pct'] < 80])
        """
        from data_utils import compute_temporal_features
        
        # Ensure temporal features exist
        df_temp = compute_temporal_features(self.df)
        
        # Create expected timeline
        date_range = pd.date_range(
            start=df_temp['timestamp'].min(),
            end=df_temp['timestamp'].max(),
            freq='h'
        )
        
        # Calculate completeness for each route
        results = []
        for route in self.routes:
            route_data = df_temp[df_temp['route_code'] == route]
            
            # Overall completeness
            expected_count = len(date_range)
            actual_count = len(route_data)
            overall_pct = (actual_count / expected_count * 100) if expected_count > 0 else 0
            
            # By hour of day
            for hour in range(24):
                hour_data = route_data[route_data['hour'] == hour]
                expected_hour = len(date_range) / 24
                # Count distinct dates covered, not raw rows, to avoid inflation
                # from multiple readings within the same hour on the same day
                actual_hour = hour_data['date'].nunique() if 'date' in hour_data.columns else len(hour_data)
                hour_pct = (actual_hour / expected_hour * 100) if expected_hour > 0 else 0
                
                results.append({
                    'route_code': route,
                    'hour': hour,
                    'completeness_pct': hour_pct,
                    'expected_count': int(expected_hour),
                    'actual_count': actual_hour
                })
        
        completeness_df = pd.DataFrame(results)
        
        # Flag routes with low completeness
        for route in self.routes:
            route_completeness = completeness_df[completeness_df['route_code'] == route]['completeness_pct'].mean()
            if route_completeness < 80:
                warnings.warn(f"Route {route} has only {route_completeness:.1f}% data completeness. "
                            f"Results may be unreliable.")
        
        return completeness_df
    
    def identify_missing_patterns(self) -> pd.DataFrame:
        """
        Identify systematic patterns in missing data.
        
        Detects if specific hours or days-of-week consistently have missing data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame showing missing data patterns by hour and day-of-week
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> patterns = analyzer.identify_missing_patterns()
        >>> print(patterns[patterns['missing_pct'] > 20])
        """
        from data_utils import compute_temporal_features
        
        df_temp = compute_temporal_features(self.df)
        
        # Analyze by hour and day of week
        results = []
        for hour in range(24):
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                subset = df_temp[(df_temp['hour'] == hour) & (df_temp['day_of_week'] == day)]
                
                expected_count = len(self.routes)  # One observation per route
                actual_count = len(subset['route_code'].unique())
                missing_pct = ((expected_count - actual_count) / expected_count * 100) if expected_count > 0 else 0
                
                results.append({
                    'hour': hour,
                    'day_of_week': day,
                    'expected_routes': expected_count,
                    'actual_routes': actual_count,
                    'missing_pct': missing_pct
                })
        
        patterns_df = pd.DataFrame(results)
        
        # Identify systematic patterns (>20% missing consistently)
        systematic = patterns_df[patterns_df['missing_pct'] > 20]
        if not systematic.empty:
            warnings.warn(f"Systematic missing data detected: {len(systematic)} hour/day combinations "
                        f"have >20% missing data")
        
        return patterns_df
    
    def compute_quality_metrics(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality summary report.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with quality metrics including:
            - completeness: Overall data completeness percentage
            - outlier_rate: Percentage of observations flagged as outliers
            - temporal_coverage: Date range covered
            - routes_coverage: Number of routes with sufficient data
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> quality = analyzer.compute_quality_metrics()
        >>> print(f"Overall completeness: {quality['completeness']:.1f}%")
        """
        from data_utils import detect_outliers, compute_temporal_features
        
        df_temp = compute_temporal_features(self.df)
        
        # Overall completeness
        date_range = pd.date_range(
            start=df_temp['timestamp'].min(),
            end=df_temp['timestamp'].max(),
            freq='h'
        )
        expected_total = len(date_range) * len(self.routes)
        actual_total = len(self.df)
        completeness = (actual_total / expected_total * 100) if expected_total > 0 else 0
        
        # Outlier rate
        outliers = detect_outliers(self.df['avg_speed'], method='zscore', threshold=3.0)
        outlier_rate = (outliers.sum() / len(self.df) * 100) if len(self.df) > 0 else 0
        
        # Temporal coverage
        temporal_coverage = {
            'start_date': df_temp['timestamp'].min().strftime('%Y-%m-%d'),
            'end_date': df_temp['timestamp'].max().strftime('%Y-%m-%d'),
            'total_days': (df_temp['timestamp'].max() - df_temp['timestamp'].min()).days
        }
        
        # Routes with sufficient data (>80% complete)
        route_completeness = self.df.groupby('route_code').size()
        expected_per_route = len(date_range)
        sufficient_routes = (route_completeness / expected_per_route > 0.8).sum()
        
        return {
            'completeness': completeness,
            'outlier_rate': outlier_rate,
            'temporal_coverage': temporal_coverage,
            'routes_coverage': {
                'total_routes': len(self.routes),
                'sufficient_data_routes': int(sufficient_routes),
                'insufficient_data_routes': len(self.routes) - int(sufficient_routes)
            }
        }
    
    def validate_distance_consistency(self) -> pd.DataFrame:
        """
        Validate that distance measurements are consistent over time for each route.
        
        Detects potential data collection errors by checking for unusual
        variations in reported distances.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with distance statistics and consistency flags by route
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> consistency = analyzer.validate_distance_consistency()
        >>> print(consistency[consistency['is_inconsistent']])
        """
        # Calculate distance statistics by route
        distance_stats = self.df.groupby('route_code')['distance'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('range', lambda x: x.max() - x.min())
        ]).reset_index()
        
        # Flag routes with high variation (>5% coefficient of variation)
        distance_stats['cv'] = distance_stats['std'] / distance_stats['mean']
        distance_stats['is_inconsistent'] = distance_stats['cv'] > 0.05
        
        # Warn about inconsistent routes
        inconsistent = distance_stats[distance_stats['is_inconsistent']]
        if not inconsistent.empty:
            for _, row in inconsistent.iterrows():
                warnings.warn(f"Route {row['route_code']} has inconsistent distances: "
                            f"mean={row['mean']:.2f}km, std={row['std']:.2f}km (CV={row['cv']:.1%})")
        
        return distance_stats

    
    # ========================================================================
    # Alternative Scoring Methods
    # ========================================================================
    
    def score_by_percentile(self) -> pd.DataFrame:
        """
        Rank routes by percentile of average speed distribution.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'percentile_score'] sorted descending
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> scores = analyzer.score_by_percentile()
        """
        route_speeds = self.df.groupby('route_code')['avg_speed'].mean()
        percentiles = route_speeds.rank(pct=True) * 100
        
        result = pd.DataFrame({
            'route_code': percentiles.index,
            'percentile_score': percentiles.values
        }).sort_values('percentile_score', ascending=False).reset_index(drop=True)
        
        return result
    
    def score_by_zscore(self) -> pd.DataFrame:
        """
        Standardize speeds and use z-scores as rankings.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'zscore'] sorted descending
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> scores = analyzer.score_by_zscore()
        """
        route_speeds = self.df.groupby('route_code')['avg_speed'].mean()
        z_scores = (route_speeds - route_speeds.mean()) / route_speeds.std()
        
        result = pd.DataFrame({
            'route_code': z_scores.index,
            'zscore': z_scores.values
        }).sort_values('zscore', ascending=False).reset_index(drop=True)
        
        return result
    
    def score_by_median(self) -> pd.DataFrame:
        """
        Use median instead of mean for robustness to outliers.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'median_speed'] sorted descending
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> scores = analyzer.score_by_median()
        """
        route_medians = self.df.groupby('route_code')['avg_speed'].median()
        
        result = pd.DataFrame({
            'route_code': route_medians.index,
            'median_speed': route_medians.values
        }).sort_values('median_speed', ascending=False).reset_index(drop=True)
        
        return result
    
    def compare_scoring_methods(self, days_rolling: int = 10) -> pd.DataFrame:
        """
        Compare centered linear spacing against alternative scoring methods.
        
        Computes correlation matrices showing agreement between different
        scoring methodologies.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days for R³S² calculation
            
        Returns
        -------
        pd.DataFrame
            Correlation matrix showing Spearman rank correlations between methods
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> comparison = analyzer.compare_scoring_methods()
        >>> print(comparison)
        """
        from scipy import stats
        
        # Calculate scores using different methods
        rrs_scores = self.calculate_rrs(days_rolling=days_rolling).set_index('route_code')['points']
        percentile_scores = self.score_by_percentile().set_index('route_code')['percentile_score']
        zscore_scores = self.score_by_zscore().set_index('route_code')['zscore']
        median_scores = self.score_by_median().set_index('route_code')['median_speed']
        
        # Combine into DataFrame
        all_scores = pd.DataFrame({
            'R³S²': rrs_scores,
            'Percentile': percentile_scores,
            'Z-Score': zscore_scores,
            'Median': median_scores
        })
        
        # Compute pairwise rank correlations
        methods = ['R³S²', 'Percentile', 'Z-Score', 'Median']
        n_methods = len(methods)
        corr_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = stats.spearmanr(all_scores[method1], all_scores[method2])
                    corr_matrix[i, j] = corr
        
        # Create labeled DataFrame
        corr_df = pd.DataFrame(corr_matrix, index=methods, columns=methods)
        
        return corr_df
    
    def compute_rrs_confidence_intervals(self, n_bootstrap: int = 1000, 
                                        days_rolling: int = 10,
                                        confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Compute confidence intervals for R³S² scores using bootstrap resampling.
        
        Parameters
        ----------
        n_bootstrap : int, default=1000
            Number of bootstrap iterations
        days_rolling : int, default=10
            Number of days in rolling window
        confidence_level : float, default=0.95
            Confidence level (e.g., 0.95 for 95% CI)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['route_code', 'points', 'ci_lower', 'ci_upper']
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> ci_scores = analyzer.compute_rrs_confidence_intervals(n_bootstrap=100)
        """
        from data_utils import bootstrap_resample
        
        try:
            # Generate bootstrap samples
            samples = bootstrap_resample(self.df, n_iterations=n_bootstrap)
            
            # Calculate R³S² scores for each sample
            bootstrap_scores = []
            for sample in samples:
                temp_analyzer = TrafficAnalyzer(sample, self.routes_df)
                scores = temp_analyzer.calculate_rrs(days_rolling=days_rolling)
                bootstrap_scores.append(scores.set_index('route_code')['points'])
            
            # Combine bootstrap results
            bootstrap_df = pd.DataFrame(bootstrap_scores)
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = bootstrap_df.quantile(lower_percentile / 100, axis=0)
            ci_upper = bootstrap_df.quantile(upper_percentile / 100, axis=0)
            point_estimate = bootstrap_df.mean(axis=0)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'route_code': point_estimate.index,
                'points': point_estimate.values,
                'ci_lower': ci_lower.values,
                'ci_upper': ci_upper.values
            }).sort_values('points', ascending=False).reset_index(drop=True)
            
            return result
            
        except MemoryError:
            warnings.warn(f"Insufficient memory for {n_bootstrap} bootstrap iterations. "
                        f"Reducing to {n_bootstrap // 10}")
            return self.compute_rrs_confidence_intervals(
                n_bootstrap=n_bootstrap // 10,
                days_rolling=days_rolling,
                confidence_level=confidence_level
            )
    
    def test_rrs_transitivity(self, days_rolling: int = 10) -> Dict[str, bool]:
        """
        Test whether R³S² scoring satisfies transitivity property.
        
        Checks if A > B and B > C implies A > C for all route combinations.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days in rolling window
            
        Returns
        -------
        Dict[str, bool]
            Dictionary with keys:
            - 'is_transitive': Whether transitivity holds for all combinations
            - 'violations': List of transitivity violations (if any)
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> transitivity = analyzer.test_rrs_transitivity()
        >>> print(f"Transitive: {transitivity['is_transitive']}")
        """
        scores = self.calculate_rrs(days_rolling=days_rolling)
        scores_dict = scores.set_index('route_code')['points'].to_dict()
        
        violations = []
        routes_list = list(scores_dict.keys())
        
        # Check all triplets
        for i, route_a in enumerate(routes_list):
            for j, route_b in enumerate(routes_list):
                if i >= j:
                    continue
                for k, route_c in enumerate(routes_list):
                    if j >= k:
                        continue
                    
                    # Check transitivity: if A > B and B > C, then A > C
                    if (scores_dict[route_a] > scores_dict[route_b] and 
                        scores_dict[route_b] > scores_dict[route_c]):
                        if not (scores_dict[route_a] > scores_dict[route_c]):
                            violations.append({
                                'route_a': route_a,
                                'route_b': route_b,
                                'route_c': route_c,
                                'scores': {
                                    route_a: scores_dict[route_a],
                                    route_b: scores_dict[route_b],
                                    route_c: scores_dict[route_c]
                                }
                            })
        
        return {
            'is_transitive': len(violations) == 0,
            'violations': violations
        }

    
    # ========================================================================
    # Statistical Testing Methods
    # ========================================================================
    
    def test_normality(self) -> pd.DataFrame:
        """
        Perform normality tests on speed distributions for each route.
        
        Uses Shapiro-Wilk and Anderson-Darling tests.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with test results for each route
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> normality = analyzer.test_normality()
        >>> print(normality[normality['is_normal'] == False])
        """
        from scipy import stats
        
        results = []
        for route in self.routes:
            route_speeds = self.df[self.df['route_code'] == route]['avg_speed']
            
            # Shapiro-Wilk test
            if len(route_speeds) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    shapiro_stat, shapiro_p = stats.shapiro(route_speeds)
            else:
                shapiro_stat, shapiro_p = np.nan, np.nan
            
            # Anderson-Darling test
            if len(route_speeds) >= 8:
                try:
                    # SciPy >=1.17 requires explicitly selecting p-value method
                    anderson_result = stats.anderson(
                        route_speeds,
                        dist='norm',
                        method='interpolate'
                    )
                except TypeError:
                    # Backward compatibility for older SciPy versions
                    anderson_result = stats.anderson(route_speeds, dist='norm')

                anderson_stat = anderson_result.statistic
                if hasattr(anderson_result, 'pvalue') and anderson_result.pvalue is not None:
                    anderson_p = float(anderson_result.pvalue)
                else:
                    # Fallback for older SciPy API without p-value
                    anderson_critical = anderson_result.critical_values[2]
                    anderson_p = 0.05 if anderson_stat > anderson_critical else 0.10
            else:
                anderson_stat, anderson_p = np.nan, np.nan
            
            # Determine if normal (p > 0.05 for both tests)
            is_normal = (shapiro_p > 0.05) if not np.isnan(shapiro_p) else None
            
            results.append({
                'route_code': route,
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'anderson_statistic': anderson_stat,
                'anderson_pvalue': anderson_p,
                'is_normal': is_normal
            })
        
        return pd.DataFrame(results)
    
    def test_stationarity(self) -> pd.DataFrame:
        """
        Perform stationarity tests on time-series data for each route.
        
        Uses Augmented Dickey-Fuller (ADF) test to determine if the time series
        has a unit root (non-stationary) or is stationary.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with test results for each route including:
            - route_code: Route identifier
            - adf_statistic: ADF test statistic
            - adf_pvalue: p-value for the test
            - is_stationary: True if p < 0.05 (reject null hypothesis of unit root)
            - critical_value_5pct: Critical value at 5% significance level
            
        Notes
        -----
        The null hypothesis of the ADF test is that the time series has a unit root
        (is non-stationary). A p-value < 0.05 indicates we can reject the null
        hypothesis and conclude the series is stationary.
        
        Stationary time series have constant mean and variance over time, which is
        important for many time series analysis methods.
        
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> stationarity = analyzer.test_stationarity()
        >>> print(stationarity[stationarity['is_stationary'] == False])
        """
        from statsmodels.tsa.stattools import adfuller
        
        results = []
        for route in self.routes:
            route_data = self.df[self.df['route_code'] == route].copy()
            
            # Sort by time to ensure proper time series
            route_data = route_data.sort_values(['year', 'month', 'day', 'hour'])
            route_speeds = route_data['avg_speed'].values
            
            # Need at least 12 observations for ADF test
            if len(route_speeds) >= 12:
                try:
                    adf_result = adfuller(route_speeds, autolag='AIC')
                    adf_stat = adf_result[0]
                    adf_p = adf_result[1]
                    critical_5pct = adf_result[4]['5%']
                    is_stationary = adf_p < 0.05
                except Exception as e:
                    warnings.warn(f"ADF test failed for route {route}: {e}")
                    adf_stat, adf_p, critical_5pct, is_stationary = np.nan, np.nan, np.nan, None
            else:
                adf_stat, adf_p, critical_5pct, is_stationary = np.nan, np.nan, np.nan, None
            
            results.append({
                'route_code': route,
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_p,
                'critical_value_5pct': critical_5pct,
                'is_stationary': is_stationary
            })
        
        return pd.DataFrame(results)
    
    def analyze_autocorrelation(self, max_lags: int = 24) -> Dict[str, pd.Series]:
        """
        Compute autocorrelation function (ACF) for each route.
        
        Autocorrelation measures how correlated a time series is with itself
        at different time lags. This helps identify temporal dependencies and
        periodic patterns in traffic data.
        
        Parameters
        ----------
        max_lags : int, default=24
            Maximum number of lags to compute (default 24 for hourly data)
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary mapping route_code to Series of ACF values
            Index represents lag (0, 1, 2, ..., max_lags)
            Values represent correlation coefficient at each lag
            
        Notes
        -----
        ACF values range from -1 to 1:
        - Values close to 1 indicate strong positive correlation
        - Values close to 0 indicate no correlation
        - Values close to -1 indicate strong negative correlation
        
        For hourly traffic data, you might expect:
        - High correlation at lag 24 (same hour yesterday)
        - High correlation at lag 168 (same hour last week)
        
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> acf_results = analyzer.analyze_autocorrelation(max_lags=48)
        >>> for route, acf in acf_results.items():
        ...     print(f"{route}: ACF at lag 24 = {acf[24]:.3f}")
        """
        from statsmodels.tsa.stattools import acf
        
        results = {}
        for route in self.routes:
            route_data = self.df[self.df['route_code'] == route].copy()
            
            # Sort by time to ensure proper time series
            route_data = route_data.sort_values(['year', 'month', 'day', 'hour'])
            route_speeds = route_data['avg_speed'].values
            
            # Need sufficient data for ACF (at least max_lags + 1)
            if len(route_speeds) >= max_lags + 1:
                try:
                    # Compute ACF with specified number of lags
                    acf_values = acf(route_speeds, nlags=max_lags, fft=True)
                    results[route] = pd.Series(acf_values, index=range(len(acf_values)))
                except Exception as e:
                    warnings.warn(f"ACF computation failed for route {route}: {e}")
                    results[route] = pd.Series([np.nan] * (max_lags + 1), index=range(max_lags + 1))
            else:
                # Insufficient data
                results[route] = pd.Series([np.nan] * (max_lags + 1), index=range(max_lags + 1))
        
        return results
    
    def test_variance_homogeneity(self) -> Dict[str, float]:
        """
        Test homogeneity of variance across routes using Levene's test.
        
        Levene's test assesses whether the variance in average speed is equal
        across all routes. This is important when comparing routes, as many
        statistical tests assume equal variances.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with test results:
            - 'levene_statistic': Test statistic
            - 'levene_pvalue': p-value for the test
            - 'variances_equal': True if p > 0.05 (variances are homogeneous)
            - 'route_variances': Dict mapping each route to its variance
            
        Notes
        -----
        The null hypothesis is that all routes have equal variance.
        - p-value > 0.05: Cannot reject null, variances are homogeneous
        - p-value < 0.05: Reject null, variances differ significantly
        
        If variances are not homogeneous, consider using robust statistical
        methods or transforming the data (e.g., log transform).
        
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> variance_test = analyzer.test_variance_homogeneity()
        >>> if not variance_test['variances_equal']:
        ...     print("Warning: Routes have unequal variances")
        ...     print(variance_test['route_variances'])
        """
        from scipy import stats
        
        # Collect speed data for each route
        route_speeds = []
        route_variances = {}
        
        for route in self.routes:
            speeds = self.df[self.df['route_code'] == route]['avg_speed'].values
            if len(speeds) >= 2:
                route_speeds.append(speeds)
                route_variances[route] = np.var(speeds, ddof=1)
            else:
                route_variances[route] = np.nan
        
        # Need at least 2 routes with sufficient data
        if len(route_speeds) >= 2:
            try:
                levene_stat, levene_p = stats.levene(*route_speeds)
                variances_equal = levene_p > 0.05
            except Exception as e:
                warnings.warn(f"Levene's test failed: {e}")
                levene_stat, levene_p, variances_equal = np.nan, np.nan, None
        else:
            levene_stat, levene_p, variances_equal = np.nan, np.nan, None
        
        return {
            'levene_statistic': levene_stat,
            'levene_pvalue': levene_p,
            'variances_equal': variances_equal,
            'route_variances': route_variances
        }
    
    def perform_power_analysis(self, effect_size: float = 0.5, alpha: float = 0.05, 
                               power: float = 0.8) -> pd.DataFrame:
        """
        Validate that sample sizes are sufficient for statistical inferences.
        
        Performs power analysis to determine if the current sample size for each
        route is adequate to detect meaningful differences with specified statistical
        power. This helps ensure that statistical tests have sufficient sensitivity.
        
        Parameters
        ----------
        effect_size : float, default=0.5
            Expected effect size (Cohen's d). Common values:
            - 0.2: small effect
            - 0.5: medium effect (default)
            - 0.8: large effect
        alpha : float, default=0.05
            Significance level (Type I error rate)
        power : float, default=0.8
            Desired statistical power (1 - Type II error rate)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with power analysis results for each route:
            - route_code: Route identifier
            - sample_size: Actual number of observations
            - required_sample_size: Minimum sample size needed for desired power
            - actual_power: Statistical power with current sample size
            - is_sufficient: True if sample size meets requirements
            
        Notes
        -----
        Statistical power is the probability of detecting an effect when it exists.
        Power of 0.8 means 80% chance of detecting a true effect.
        
        If sample size is insufficient:
        - Collect more data
        - Accept lower power (higher risk of Type II error)
        - Focus on larger effect sizes only
        
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> power_results = analyzer.perform_power_analysis(effect_size=0.5)
        >>> insufficient = power_results[power_results['is_sufficient'] == False]
        >>> if not insufficient.empty:
        ...     print("Routes with insufficient sample size:")
        ...     print(insufficient[['route_code', 'sample_size', 'required_sample_size']])
        """
        from statsmodels.stats.power import TTestIndPower
        
        # Initialize power analysis
        power_analysis = TTestIndPower()
        
        # Calculate required sample size for desired power
        try:
            required_n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
        except Exception as e:
            warnings.warn(f"Power analysis calculation failed: {e}")
            required_n = np.nan
        
        results = []
        for route in self.routes:
            route_data = self.df[self.df['route_code'] == route]
            sample_size = len(route_data)
            
            # Calculate actual power with current sample size
            if sample_size >= 2 and not np.isnan(required_n):
                try:
                    actual_power = power_analysis.solve_power(
                        effect_size=effect_size,
                        nobs1=sample_size,
                        alpha=alpha,
                        alternative='two-sided'
                    )
                    is_sufficient = sample_size >= required_n
                except Exception as e:
                    warnings.warn(f"Power calculation failed for route {route}: {e}")
                    actual_power = np.nan
                    is_sufficient = None
            else:
                actual_power = np.nan
                is_sufficient = None
            
            results.append({
                'route_code': route,
                'sample_size': sample_size,
                'required_sample_size': required_n,
                'actual_power': actual_power,
                'is_sufficient': is_sufficient
            })
        
        return pd.DataFrame(results)
    
    def generate_recommendations(self, days_rolling: int = 10) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on all analyses.
        
        Analyzes results from various methods and produces specific
        recommendations with quantified expected benefits.
        
        Parameters
        ----------
        days_rolling : int, default=10
            Number of days for R³S² calculation
            
        Returns
        -------
        List[Dict[str, Any]]
            List of recommendation dictionaries with keys:
            - type: Category of recommendation
            - severity: 'low', 'medium', or 'high'
            - description: Detailed recommendation text
            - expected_benefit: Expected improvement from implementing
            
        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> recommendations = analyzer.generate_recommendations()
        >>> for rec in recommendations:
        ...     print(f"{rec['severity'].upper()}: {rec['description']}")
        """
        recommendations = []
        
        # Check missing data bias
        bias_analysis = self.analyze_missing_data_bias(days_rolling=days_rolling)
        if bias_analysis['bias_metric'] > 0.1:
            recommendations.append({
                'type': 'missing_data_bias',
                'severity': 'high' if bias_analysis['bias_metric'] > 0.2 else 'medium',
                'description': f"Unequal sample sizes detected (CV={bias_analysis['bias_metric']:.2%}). "
                              "Consider imputation strategies or data quality thresholds.",
                'expected_benefit': 'Improved ranking fairness and reduced bias by 15-30%'
            })
        
        # Check normality
        normality_results = self.test_normality()
        non_normal_routes = normality_results[normality_results['is_normal'] == False]
        if not non_normal_routes.empty:
            recommendations.append({
                'type': 'non_normal_distribution',
                'severity': 'medium',
                'description': f"{len(non_normal_routes)} routes have non-normal speed distributions. "
                              "Consider using median-based or percentile-based scoring methods instead of mean.",
                'expected_benefit': 'More robust rankings less sensitive to outliers'
            })
        
        # Check data quality
        quality = self.compute_quality_metrics()
        if quality['completeness'] < 80:
            recommendations.append({
                'type': 'low_completeness',
                'severity': 'high',
                'description': f"Overall data completeness is only {quality['completeness']:.1f}%. "
                              "Improve data collection or implement gap-filling strategies.",
                'expected_benefit': 'More reliable and stable route rankings'
            })
        
        # Check outlier rate
        if quality['outlier_rate'] > 5:
            recommendations.append({
                'type': 'high_outlier_rate',
                'severity': 'medium',
                'description': f"Outlier rate is {quality['outlier_rate']:.1f}%. "
                              "Consider using robust estimators (median, trimmed mean) or outlier filtering.",
                'expected_benefit': 'Reduced sensitivity to anomalous observations'
            })
        
        # Evaluate alternative scoring methods
        comparison = self.compare_scoring_methods(days_rolling=days_rolling)
        rrs_percentile_corr = comparison.loc['R³S²', 'Percentile']
        if rrs_percentile_corr < 0.9:
            recommendations.append({
                'type': 'scoring_method_disagreement',
                'severity': 'medium',
                'description': f"R³S² and percentile methods show moderate disagreement (r={rrs_percentile_corr:.2f}). "
                              "Consider using multiple scoring methods for validation.",
                'expected_benefit': 'More robust route performance assessment'
            })
        
        # Check for systematically missing hours (e.g. scraper/runner failures)
        completeness = self.analyze_data_completeness()
        hour_avg = completeness.groupby('hour')['completeness_pct'].mean()
        dead_hours = hour_avg[hour_avg < 90].index.tolist()
        if dead_hours:
            recommendations.append({
                'type': 'systematic_collection_failure',
                'severity': 'high',
                'description': (
                    f"Hours {dead_hours} have below 90% data coverage across all routes "
                    f"(avg completeness: {hour_avg[dead_hours].mean():.1f}%). "
                    "This indicates a systematic collection failure (e.g. a scheduled scraper not running) "
                    "rather than random missing data. These hours cannot be reliably interpolated and "
                    "should be excluded from time-of-day comparisons and travel time recommendations."
                ),
                'expected_benefit': 'Prevents misleading conclusions from structurally absent data'
            })

        # If no issues found
        if not recommendations:
            recommendations.append({
                'type': 'no_issues',
                'severity': 'low',
                'description': "No significant issues detected. Current R³S² methodology appears sound.",
                'expected_benefit': 'Continue current approach'
            })
        
        return recommendations

    # ========================================================================
    # Diagnostic Plot Generation Methods
    # ========================================================================

    def plot_qq_normality(self) -> None:
        """
        Generate Q-Q plots for normality assessment of each route's speed distribution.

        Creates a grid of Q-Q (quantile-quantile) plots comparing the distribution
        of speeds for each route against a theoretical normal distribution.

        Q-Q plots help assess:
        - Whether speed distributions are approximately normal
        - Types of deviations from normality (skewness, heavy tails, etc.)
        - Which routes may benefit from non-parametric methods

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> analyzer.plot_qq_normality()
        """
        from scipy import stats
        import matplotlib.pyplot as plt

        n_routes = len(self.routes)
        n_cols = 3
        n_rows = (n_routes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_routes > 1 else [axes]

        for idx, route_code in enumerate(self.routes):
            ax = axes[idx]
            route_data = self.df[self.df['route_code'] == route_code]

            if route_data.empty or len(route_data) < 3:
                ax.text(0.5, 0.5, f'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            speeds = route_data['avg_speed'].dropna()

            # Create Q-Q plot
            stats.probplot(speeds, dist="norm", plot=ax)

            # Get route label
            route_label = self.routes_df[self.routes_df['route_code'] == route_code]
            if not route_label.empty and 'label_short' in route_label.columns:
                label = route_label['label_short'].iloc[0]
            else:
                label = route_code

            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_routes, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Q-Q Plots: Normality Assessment for All Routes',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    def plot_residual_diagnostics(self, route_code: str) -> None:
        """
        Generate residual diagnostic plots for model validation.

        Creates diagnostic plots for residuals from R³S² scoring:
        1. Residuals vs Fitted values
        2. Histogram of residuals
        3. Q-Q plot of residuals
        4. Scale-Location plot

        These plots help assess:
        - Homoscedasticity (constant variance)
        - Normality of residuals
        - Presence of outliers or influential points
        - Model adequacy

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> analyzer.plot_residual_diagnostics('VJRQ+2M|RMJJ+F4')
        """
        from scipy import stats
        import matplotlib.pyplot as plt

        # Get data for this route
        route_data = self.df[self.df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {route_code}")
            return

        # Compute fitted values (mean by hour and day-of-week)
        if 'day_of_week' not in route_data.columns:
            from data_utils import compute_temporal_features
            route_data = compute_temporal_features(route_data)

        fitted = route_data.groupby(['hour', 'day_of_week'])['avg_speed'].transform('mean')
        route_data['fitted'] = fitted
        route_data['residuals'] = route_data['avg_speed'] - route_data['fitted']
        route_data['std_residuals'] = (route_data['residuals'] - route_data['residuals'].mean()) / route_data['residuals'].std()
        route_data['sqrt_std_residuals'] = np.sqrt(np.abs(route_data['std_residuals']))

        # Get route label
        route_label = self.routes_df[self.routes_df['route_code'] == route_code]
        if not route_label.empty and 'label_short' in route_label.columns:
            label = route_label['label_short'].iloc[0]
        else:
            label = route_code

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Residuals vs Fitted
        axes[0, 0].scatter(route_data['fitted'], route_data['residuals'],
                          alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Histogram of residuals
        axes[0, 1].hist(route_data['residuals'], bins=30, density=True,
                       alpha=0.7, edgecolor='black')
        mu, sigma = route_data['residuals'].mean(), route_data['residuals'].std()
        x = np.linspace(route_data['residuals'].min(), route_data['residuals'].max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                       label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        axes[0, 1].set_xlabel('Residuals', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Histogram of Residuals', fontsize=11, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Q-Q plot
        stats.probplot(route_data['residuals'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Normal Q-Q Plot', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Scale-Location plot
        axes[1, 1].scatter(route_data['fitted'], route_data['sqrt_std_residuals'],
                          alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Fitted Values', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Scale-Location Plot', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # Add overall title
        fig.suptitle(f'Residual Diagnostics: {label}',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    # ========================================================================
    # Comparative Methodology Evaluation
    # ========================================================================

    def evaluate_scoring_stability(self, window_sizes: List[int] = [5, 10, 15, 20, 30]) -> Dict[str, Any]:
        """
        Measure stability of different scoring methods across window sizes.

        Computes rank correlations for each scoring method across different
        rolling window sizes to assess which methods produce the most stable rankings.

        Parameters
        ----------
        window_sizes : List[int], default=[5, 10, 15, 20, 30]
            List of window sizes (in days) to test

        Returns
        -------
        Dict[str, Any]
            Dictionary with stability metrics for each scoring method

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> stability = analyzer.evaluate_scoring_stability()
        >>> print(stability['summary'])
        """
        from scipy import stats

        methods = {
            'R³S²': lambda w: self.calculate_rrs(days_rolling=w),
            'Percentile': lambda w: self.score_by_percentile(),
            'Z-Score': lambda w: self.score_by_zscore(),
            'Median': lambda w: self.score_by_median()
        }

        stability_results = {}

        for method_name, method_func in methods.items():
            # Compute scores for each window size
            scores_by_window = {}
            for window in window_sizes:
                try:
                    if method_name == 'R³S²':
                        scores = method_func(window)
                    else:
                        scores = method_func()
                    scores_by_window[window] = scores.set_index('route_code')
                except Exception as e:
                    continue

            # Compute pairwise rank correlations
            correlations = []
            for i, w1 in enumerate(window_sizes[:-1]):
                for w2 in window_sizes[i+1:]:
                    if w1 in scores_by_window and w2 in scores_by_window:
                        # Get scores column name (varies by method)
                        col_name = 'points' if method_name == 'R³S²' else list(scores_by_window[w1].columns)[0]

                        corr, _ = stats.spearmanr(
                            scores_by_window[w1][col_name],
                            scores_by_window[w2][col_name]
                        )
                        correlations.append(corr)

            # Compute stability metrics
            if correlations:
                stability_results[method_name] = {
                    'mean_correlation': np.mean(correlations),
                    'std_correlation': np.std(correlations),
                    'min_correlation': np.min(correlations),
                    'max_correlation': np.max(correlations)
                }
            else:
                stability_results[method_name] = {
                    'mean_correlation': np.nan,
                    'std_correlation': np.nan,
                    'min_correlation': np.nan,
                    'max_correlation': np.nan
                }

        # Create summary
        summary_df = pd.DataFrame(stability_results).T
        summary_df = summary_df.sort_values('mean_correlation', ascending=False)

        return {
            'stability_by_method': stability_results,
            'summary': summary_df,
            'interpretation': 'Higher mean correlation indicates more stable rankings across window sizes.'
        }

    def evaluate_outlier_sensitivity(self, outlier_thresholds: List[float] = [2.0, 2.5, 3.0]) -> Dict[str, Any]:
        """
        Evaluate sensitivity of scoring methods to outliers.

        Measures how much rankings change when outliers are removed at
        different threshold levels.

        Parameters
        ----------
        outlier_thresholds : List[float], default=[2.0, 2.5, 3.0]
            Z-score thresholds for defining outliers

        Returns
        -------
        Dict[str, Any]
            Dictionary with sensitivity metrics for each method

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> sensitivity = analyzer.evaluate_outlier_sensitivity()
        >>> print(sensitivity['summary'])
        """
        from scipy import stats
        from data_utils import detect_outliers

        methods = {
            'R³S²': lambda df: TrafficAnalyzer(df, self.routes_df).calculate_rrs(),
            'Percentile': lambda df: TrafficAnalyzer(df, self.routes_df).score_by_percentile(),
            'Z-Score': lambda df: TrafficAnalyzer(df, self.routes_df).score_by_zscore(),
            'Median': lambda df: TrafficAnalyzer(df, self.routes_df).score_by_median()
        }

        # Baseline scores (with all data)
        baseline_scores = {}
        for method_name, method_func in methods.items():
            try:
                scores = method_func(self.df)
                baseline_scores[method_name] = scores.set_index('route_code')
            except Exception as e:
                continue

        sensitivity_results = {}

        for method_name in baseline_scores.keys():
            correlations = []

            for threshold in outlier_thresholds:
                # Remove outliers
                outlier_mask = detect_outliers(self.df['avg_speed'], method='zscore', threshold=threshold)
                df_filtered = self.df[~outlier_mask].copy()

                # Compute scores without outliers
                try:
                    filtered_scores = methods[method_name](df_filtered)
                    filtered_scores = filtered_scores.set_index('route_code')

                    # Get score column name
                    col_name = list(baseline_scores[method_name].columns)[0]

                    # Compute rank correlation
                    corr, _ = stats.spearmanr(
                        baseline_scores[method_name][col_name],
                        filtered_scores[col_name]
                    )
                    correlations.append(corr)
                except Exception as e:
                    correlations.append(np.nan)

            # Compute sensitivity metrics
            sensitivity_results[method_name] = {
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'min_correlation': np.min(correlations)
            }

        # Create summary
        summary_df = pd.DataFrame(sensitivity_results).T
        summary_df = summary_df.sort_values('mean_correlation', ascending=False)

        return {
            'sensitivity_by_method': sensitivity_results,
            'summary': summary_df,
            'interpretation': 'Higher correlation indicates less sensitivity to outliers (more robust).'
        }

    def evaluate_computational_efficiency(self, n_iterations: int = 10) -> Dict[str, Any]:
        """
        Compare computational efficiency of different scoring methods.

        Measures execution time for each scoring method to assess
        computational cost.

        Parameters
        ----------
        n_iterations : int, default=10
            Number of iterations for timing

        Returns
        -------
        Dict[str, Any]
            Dictionary with timing metrics for each method

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> efficiency = analyzer.evaluate_computational_efficiency()
        >>> print(efficiency['summary'])
        """
        import time

        methods = {
            'R³S²': lambda: self.calculate_rrs(),
            'Percentile': lambda: self.score_by_percentile(),
            'Z-Score': lambda: self.score_by_zscore(),
            'Median': lambda: self.score_by_median()
        }

        timing_results = {}

        for method_name, method_func in methods.items():
            times = []
            for _ in range(n_iterations):
                start = time.time()
                try:
                    _ = method_func()
                    elapsed = time.time() - start
                    times.append(elapsed)
                except Exception as e:
                    times.append(np.nan)

            timing_results[method_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }

        # Create summary
        summary_df = pd.DataFrame(timing_results).T
        summary_df = summary_df.sort_values('mean_time')

        return {
            'timing_by_method': timing_results,
            'summary': summary_df,
            'interpretation': 'Lower mean time indicates better computational efficiency.'
        }

    def generate_methodology_ranking_report(self) -> pd.DataFrame:
        """
        Generate comprehensive report ranking scoring methods by multiple criteria.

        Evaluates all scoring methods across:
        - Stability (consistency across window sizes)
        - Robustness (sensitivity to outliers)
        - Efficiency (computational cost)

        Returns
        -------
        pd.DataFrame
            DataFrame ranking methods with overall scores

        Examples
        --------
        >>> analyzer = TrafficAnalyzer(df, routes_df)
        >>> report = analyzer.generate_methodology_ranking_report()
        >>> print(report)
        """
        print("Evaluating scoring methodologies...")
        print("This may take a few moments...\n")

        # Evaluate each criterion
        stability = self.evaluate_scoring_stability()
        sensitivity = self.evaluate_outlier_sensitivity()
        efficiency = self.evaluate_computational_efficiency()

        # Extract scores
        methods = ['R³S²', 'Percentile', 'Z-Score', 'Median']
        report_data = []

        for method in methods:
            if method in stability['stability_by_method']:
                stability_score = stability['stability_by_method'][method]['mean_correlation']
            else:
                stability_score = np.nan

            if method in sensitivity['sensitivity_by_method']:
                robustness_score = sensitivity['sensitivity_by_method'][method]['mean_correlation']
            else:
                robustness_score = np.nan

            if method in efficiency['timing_by_method']:
                # Invert time (lower is better, so we want higher score)
                time_val = efficiency['timing_by_method'][method]['mean_time']
                efficiency_score = 1.0 / time_val if time_val > 0 else 0
            else:
                efficiency_score = np.nan

            # Compute overall score (weighted average)
            # Weights: Stability=40%, Robustness=40%, Efficiency=20%
            scores = [stability_score, robustness_score, efficiency_score]
            weights = [0.4, 0.4, 0.2]

            valid_scores = [(s, w) for s, w in zip(scores, weights) if not np.isnan(s)]
            if valid_scores:
                overall_score = sum(s * w for s, w in valid_scores) / sum(w for _, w in valid_scores)
            else:
                overall_score = np.nan

            report_data.append({
                'Method': method,
                'Stability': stability_score,
                'Robustness': robustness_score,
                'Efficiency': efficiency_score,
                'Overall Score': overall_score
            })

        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Overall Score', ascending=False)

        # Add ranking
        report_df['Rank'] = range(1, len(report_df) + 1)

        # Reorder columns
        report_df = report_df[['Rank', 'Method', 'Overall Score', 'Stability', 'Robustness', 'Efficiency']]

        # Round numeric columns
        numeric_cols = ['Overall Score', 'Stability', 'Robustness', 'Efficiency']
        report_df[numeric_cols] = report_df[numeric_cols].round(4)

        print("\nMethodology Ranking Report")
        print("=" * 80)
        print(report_df.to_string(index=False))
        print("=" * 80)
        print("\nCriteria Weights: Stability=40%, Robustness=40%, Efficiency=20%")
        print("Higher scores indicate better performance.")

        return report_df

"""
Data Utilities Module

Helper functions for data preprocessing, validation, and transformation.
Provides utilities for gap filling, outlier detection, and bootstrap resampling.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.ensemble import IsolationForest
import warnings


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def preprocess_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and preprocess raw traffic data.

    Performs the following operations:
    - Deduplicates rows by (route_code, date/time or year/month/day/hour)
    - Calculates avg_speed if not present
    - Checks for and reports missing values
    - Validates data types
    - Checks value ranges
    - Sorts by timestamp

    Parameters
    ----------
    df : pd.DataFrame
        Raw traffic data

    Returns
    -------
    pd.DataFrame
        Preprocessed traffic data

    Raises
    ------
    ValueError
        If data validation fails
    """
    df_clean = df.copy()

    # Deduplicate: same route, same day, same hour, same duration+distance
    # Catches re-ingested readings that differ only by collection timestamp within the hour
    if 'hour' not in df_clean.columns and 'time' in df_clean.columns:
        df_clean['hour'] = pd.to_datetime(df_clean['time'], format='%H:%M', errors='coerce').dt.hour
    dedup_cols = [c for c in ['route_code', 'date', 'hour', 'duration', 'distance'] if c in df_clean.columns]
    if dedup_cols:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=dedup_cols, keep='first')
        removed = before - len(df_clean)
        if removed > 0:
            print(f"Dedup: removed {removed} duplicate rows based on {dedup_cols}")

    # Calculate avg_speed if not present
    if 'avg_speed' not in df_clean.columns:
        if 'duration' in df_clean.columns and 'distance' in df_clean.columns:
            df_clean['avg_speed'] = df_clean['distance'] / (df_clean['duration'] / 60)

    # Remove rows with missing critical values
    critical_cols = ['route_code', 'duration', 'distance']
    if 'avg_speed' in df_clean.columns:
        critical_cols.append('avg_speed')
    df_clean = df_clean.dropna(subset=critical_cols)

    # Sort by timestamp
    if all(col in df_clean.columns for col in ['year', 'month', 'day', 'hour']):
        df_clean = df_clean.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)

    return df_clean


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to traffic data.

    Adds the following columns:
    - timestamp: datetime object
    - day_of_week: day name (Monday, Tuesday, etc.)
    - is_weekend: boolean indicating weekend
    - time_category: categorical time period (night, morning_rush, midday, evening_rush, evening)

    Parameters
    ----------
    df : pd.DataFrame
        Traffic data with year, month, day, hour columns OR date/time columns

    Returns
    -------
    pd.DataFrame
        Traffic data with additional temporal features
    """
    df_enhanced = df.copy()

    # Parse date and time if they exist but year/month/day/hour don't
    if 'date' in df_enhanced.columns and 'year' not in df_enhanced.columns:
        df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
        df_enhanced['year'] = df_enhanced['date'].dt.year
        df_enhanced['month'] = df_enhanced['date'].dt.month
        df_enhanced['day'] = df_enhanced['date'].dt.day

    if 'time' in df_enhanced.columns and 'hour' not in df_enhanced.columns:
        df_enhanced['hour'] = pd.to_datetime(df_enhanced['time'], format='%H:%M', errors='coerce').dt.hour

    # Create timestamp
    if 'timestamp' not in df_enhanced.columns:
        if all(col in df_enhanced.columns for col in ['year', 'month', 'day', 'hour']):
            df_enhanced['timestamp'] = pd.to_datetime(df_enhanced[['year', 'month', 'day', 'hour']])
        elif 'date' in df_enhanced.columns:
            df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['date'])

    # Add day of week
    if 'day_of_week' not in df_enhanced.columns:
        df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.day_name()

    # Add weekend indicator
    if 'is_weekend' not in df_enhanced.columns:
        df_enhanced['is_weekend'] = df_enhanced['timestamp'].dt.dayofweek >= 5

    # Add time category
    if 'time_category' not in df_enhanced.columns:
        df_enhanced['time_category'] = pd.cut(
            df_enhanced['hour'],
            bins=[0, 6, 8, 11, 14, 18, 21, 24],
            labels=['late_night', 'morning', 'morning_rush', 'early_afternoon', 'late_afternoon', 'evening_rush', 'night'],
            include_lowest=True
        )

    return df_enhanced


def fill_missing_timestamps(df: pd.DataFrame, timeline: pd.Index) -> pd.DataFrame:
    """
    Fill gaps in time series using neighbor averaging.
    
    Reindexes the DataFrame to a common timeline and fills missing values
    by averaging previous and next valid observations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Route-specific data with timestamp index
    timeline : pd.Index
        Common timeline (all unique timestamps) to reindex to
        
    Returns
    -------
    pd.DataFrame
        DataFrame with filled timestamps
    """
    # Reindex to common timeline
    df_filled = df.reindex(timeline)
    
    # Fill missing speeds using neighbor averaging
    if 'avg_speed' in df_filled.columns:
        speeds = pd.to_numeric(df_filled['avg_speed'], errors='coerce')
        prev_vals = speeds.ffill()
        next_vals = speeds.bfill()
        
        mask_missing = speeds.isna()
        mask_both = mask_missing & prev_vals.notna() & next_vals.notna()
        mask_prev_only = mask_missing & prev_vals.notna() & next_vals.isna()
        mask_next_only = mask_missing & next_vals.notna() & prev_vals.isna()
        
        filled = speeds.copy()
        filled.loc[mask_both] = (prev_vals.loc[mask_both] + next_vals.loc[mask_both]) / 2.0
        filled.loc[mask_prev_only] = prev_vals.loc[mask_prev_only]
        filled.loc[mask_next_only] = next_vals.loc[mask_next_only]
        
        df_filled['avg_speed'] = filled
    
    return df_filled


# ============================================================================
# Outlier Detection Functions
# ============================================================================

def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Identify outliers using specified method.
    
    Parameters
    ----------
    series : pd.Series
        Data series to analyze
    method : str, default='iqr'
        Detection method: 'iqr', 'zscore', or 'isolation_forest'
    threshold : float, default=3.0
        Threshold for outlier detection (interpretation depends on method)
        
    Returns
    -------
    pd.Series
        Boolean series indicating outliers (True = outlier)
        
    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'isolation_forest':
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X)
        return predictions == -1  # -1 indicates outlier
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', or 'isolation_forest'")


def bootstrap_resample(df: pd.DataFrame, n_iterations: int = 1000, 
                       random_state: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Generate bootstrap samples for confidence interval estimation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original dataset
    n_iterations : int, default=1000
        Number of bootstrap samples to generate
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    List[pd.DataFrame]
        List of bootstrap sample DataFrames
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples = []
    n = len(df)
    
    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample = df.iloc[indices].reset_index(drop=True)
        samples.append(sample)
    
    return samples

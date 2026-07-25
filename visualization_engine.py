"""
Visualization Engine Module

Comprehensive visualization generator for traffic analysis.
Provides temporal pattern visualizations, comparative performance charts,
anomaly detection plots, predictive insights, and interactive dashboard components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Optional, Any
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d


# ============================================================================
# VisualizationEngine Class
# ============================================================================

class VisualizationEngine:
    """
    Comprehensive visualization generator for traffic analysis.
    
    This class provides visualization capabilities including:
    - Temporal pattern visualizations (heatmaps, time-series decomposition)
    - Comparative performance charts (parallel coordinates, radar charts)
    - Anomaly detection visualizations (control charts, outlier plots)
    - Predictive insight visualizations (forecasts, confidence intervals)
    - Interactive dashboard components (widgets, linked plots)
    
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
        Traffic data
    routes_df : pd.DataFrame
        Route metadata
    color_palette : Dict[str, str]
        Mapping of route_code to hex color
    routes : List[str]
        List of unique route codes
        
    Examples
    --------
    >>> viz = VisualizationEngine(traffic_df, routes_df)
    >>> viz.plot_hourly_heatmap('VJRQ+2M|RMJJ+F4')
    >>> viz.plot_hour_of_day_profiles()
    """
    
    def __init__(self, df: pd.DataFrame, routes_df: pd.DataFrame):
        """
        Initialize VisualizationEngine with traffic data and route metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            Traffic data
        routes_df : pd.DataFrame
            Route metadata
        """
        # Store data
        self.df = df.copy()
        self.routes_df = routes_df.copy()
        
        # Extract unique routes
        self.routes = sorted(self.df['route_code'].dropna().unique())
        
        # Extract and store route color palette
        # If color_hex column doesn't exist, create it
        if 'color_hex' not in self.routes_df.columns:
            # Create color palette for all routes in routes_df
            routes_in_df = sorted(self.routes_df['route_code'].dropna().unique())
            palette = sns.color_palette("tab20", n_colors=len(routes_in_df))
            color_map = dict(zip(routes_in_df, [mcolors.to_hex(c) for c in palette]))
            self.routes_df['color_hex'] = self.routes_df['route_code'].map(color_map)
        
        self.color_palette = self.routes_df.set_index('route_code')['color_hex'].to_dict()
        
        # Validate that all routes have colors
        missing_colors = set(self.routes) - set(self.color_palette.keys())
        if missing_colors:
            warnings.warn(f"Routes missing color assignments: {missing_colors}")
            # Assign default colors for missing routes
            default_colors = sns.color_palette("tab20", n_colors=len(missing_colors))
            for route, color in zip(missing_colors, default_colors):
                self.color_palette[route] = mcolors.to_hex(color)
    
    def __repr__(self) -> str:
        """String representation of VisualizationEngine"""
        n_obs = len(self.df)
        n_routes = len(self.routes)
        return f"VisualizationEngine(observations={n_obs}, routes={n_routes})"
    
    def _get_route_color(self, route_code: str) -> str:
        """
        Get hex color for a route.
        
        Parameters
        ----------
        route_code : str
            Route identifier
            
        Returns
        -------
        str
            Hex color code
        """
        return self.color_palette.get(route_code, '#000000')
    
    def _apply_route_colors(self, ax: plt.Axes, route_codes: List[str]) -> None:
        """
        Apply consistent route colors to matplotlib axes.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object
        route_codes : List[str]
            List of route codes in order of appearance
        """
        lines = ax.get_lines()
        for line, route_code in zip(lines, route_codes):
            color = self._get_route_color(route_code)
            line.set_color(color)
    
    def _format_timestamp(self, timestamp: pd.Timestamp, format_str: str = '%Y-%m-%d %H:%M') -> str:
        """
        Format timestamp for display in plots.
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp to format
        format_str : str, default='%Y-%m-%d %H:%M'
            Format string for strftime
            
        Returns
        -------
        str
            Formatted timestamp string
        """
        return timestamp.strftime(format_str)
    
    def _format_hour_label(self, hour: int) -> str:
        """
        Format hour as readable label (e.g., 0 -> '12 AM', 13 -> '1 PM').
        
        Parameters
        ----------
        hour : int
            Hour in 24-hour format (0-23)
            
        Returns
        -------
        str
            Formatted hour label
        """
        if hour == 0:
            return '12 AM'
        elif hour < 12:
            return f'{hour} AM'
        elif hour == 12:
            return '12 PM'
        else:
            return f'{hour - 12} PM'
    
    def _set_figure_style(self, figsize: tuple = (12, 6), style: str = 'whitegrid') -> None:
        """
        Set consistent figure style for all plots.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 6)
            Figure size (width, height) in inches
        style : str, default='whitegrid'
            Seaborn style name
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def _get_route_label(self, route_code: str, label_type: str = 'short') -> str:
        """
        Get human-readable label for a route.
        
        Parameters
        ----------
        route_code : str
            Route identifier
        label_type : str, default='short'
            Type of label to return ('short', 'full', or 'code')
            
        Returns
        -------
        str
            Route label
        """
        if label_type == 'code':
            return route_code
        
        route_info = self.routes_df[self.routes_df['route_code'] == route_code]
        
        if route_info.empty:
            return route_code
        
        if label_type == 'short' and 'label_short' in route_info.columns:
            label = route_info['label_short'].iloc[0]
            return label if pd.notna(label) else route_code
        elif label_type == 'full' and 'label_full' in route_info.columns:
            label = route_info['label_full'].iloc[0]
            return label if pd.notna(label) else route_code
        
        return route_code
    
    def _add_statistical_significance(self, ax: plt.Axes, x1: float, x2: float, 
                                     y: float, pvalue: float, 
                                     height: float = 0.02) -> None:
        """
        Add statistical significance indicator to plot.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object
        x1, x2 : float
            X-coordinates for the comparison
        y : float
            Y-coordinate for the significance bar
        pvalue : float
            P-value from statistical test
        height : float, default=0.02
            Height of the significance bar as fraction of y-axis range
        """
        # Determine significance level
        if pvalue < 0.001:
            sig_text = '***'
        elif pvalue < 0.01:
            sig_text = '**'
        elif pvalue < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Draw horizontal line
        ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
        
        # Draw vertical ticks
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        tick_height = y_range * height
        ax.plot([x1, x1], [y, y - tick_height], 'k-', linewidth=1)
        ax.plot([x2, x2], [y, y - tick_height], 'k-', linewidth=1)
        
        # Add significance text
        ax.text((x1 + x2) / 2, y, sig_text, ha='center', va='bottom', fontsize=10)
    
    def _create_color_legend(self, ax: plt.Axes, route_codes: Optional[List[str]] = None,
                           label_type: str = 'short', loc: str = 'best') -> None:
        """
        Create a legend with route colors.
        
        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object
        route_codes : List[str], optional
            List of route codes to include in legend (default: all routes)
        label_type : str, default='short'
            Type of label to use ('short', 'full', or 'code')
        loc : str, default='best'
            Legend location
        """
        if route_codes is None:
            route_codes = self.routes
        
        # Create legend handles
        from matplotlib.patches import Patch
        handles = []
        for route_code in route_codes:
            color = self._get_route_color(route_code)
            label = self._get_route_label(route_code, label_type)
            handles.append(Patch(facecolor=color, label=label))
        
        ax.legend(handles=handles, loc=loc)
    
    def _ensure_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has temporal features (timestamp, day_of_week, etc.).
        
        Parameters
        ----------
        df : pd.DataFrame
            Traffic data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with temporal features added
        """
        from data_utils import compute_temporal_features
        
        # Check if temporal features already exist
        if 'timestamp' not in df.columns:
            df = compute_temporal_features(df)
        
        return df

    # ========================================================================
    # Temporal Pattern Visualizations
    # ========================================================================

    def plot_hourly_heatmap(self, route_code: Optional[str] = None) -> None:
        """
        Generate heatmap showing average speed by hour-of-day and day-of-week.

        Creates a heatmap visualization where:
        - X-axis: Hour of day (0-23)
        - Y-axis: Day of week (Monday-Sunday)
        - Color: Average speed (km/h)

        If route_code is None, creates a separate heatmap for each route.

        Parameters
        ----------
        route_code : str, optional
            Route identifier. If None, creates heatmaps for all routes.

        Examples
        --------
        >>> viz.plot_hourly_heatmap('VJRQ+2M|RMJJ+F4')
        >>> viz.plot_hourly_heatmap()  # All routes
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Determine which routes to plot
        routes_to_plot = [route_code] if route_code else self.routes

        # Create subplots if multiple routes
        n_routes = len(routes_to_plot)
        if n_routes == 1:
            fig, axes = plt.subplots(1, 1, figsize=(14, 6))
            axes = [axes]
        else:
            # Calculate grid dimensions
            n_cols = min(2, n_routes)
            n_rows = (n_routes + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols, 6 * n_rows))
            axes = axes.flatten() if n_routes > 1 else [axes]

        # Day of week order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for idx, route in enumerate(routes_to_plot):
            ax = axes[idx]

            # Filter data for this route
            route_data = df[df['route_code'] == route].copy()

            if route_data.empty:
                ax.text(0.5, 0.5, f'No data for {self._get_route_label(route)}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Compute average speed by hour and day of week
            pivot_data = route_data.groupby(['day_of_week', 'hour'])['avg_speed'].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='day_of_week', columns='hour', values='avg_speed')

            # Reindex to ensure all days and hours are present
            heatmap_data = heatmap_data.reindex(day_order)
            heatmap_data = heatmap_data.reindex(columns=range(24))

            # Fixed blue colormap: dark blue = slow (congested), white = fast
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                'traffic_cmap',
                ['#28609bff', '#f7f7f7ff'],  # dark blue → white
                N=256
            )

            # Create heatmap
            sns.heatmap(
                heatmap_data,
                ax=ax,
                cmap=cmap,
                annot=False,
                fmt='.1f',
                cbar_kws={'label': 'Avg Speed (km/h)'},
                linewidths=0.5,
                linecolor='white'
            )

            # Format labels
            date_min = df['timestamp'].min().strftime('%d %b %Y')
            date_max = df['timestamp'].max().strftime('%d %b %Y')
            ax.set_xlabel('Hour of Day', fontsize=11)
            ax.set_ylabel('Day of Week', fontsize=11)
            ax.set_title(
                f'Hourly Speed Heatmap: {self._get_route_label(route)}\n'
                f'{date_min} – {date_max}',
                fontsize=12, fontweight='bold'
            )

            # Format x-axis labels (show every 3 hours)
            hour_labels = [self._format_hour_label(h) if h % 3 == 0 else '' for h in range(24)]
            ax.set_xticklabels(hour_labels, rotation=0)

            # Format y-axis labels (abbreviated day names)
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.set_yticklabels(day_labels, rotation=0)

        # Hide extra subplots if any
        for idx in range(n_routes, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_calendar_heatmap(self, route_code: str) -> None:
        """
        Generate calendar heatmap showing daily average speeds with weekend indicators.

        Creates a calendar-style heatmap where:
        - Each cell represents one day
        - Color intensity shows average speed
        - Weekends are visually distinguished

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> viz.plot_calendar_heatmap('VJRQ+2M|RMJJ+F4')
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Compute daily average speed
        daily_data = route_data.groupby('timestamp').agg({
            'avg_speed': 'mean'
        }).reset_index()

        # Extract date components
        daily_data['date'] = daily_data['timestamp'].dt.date
        daily_data['year'] = daily_data['timestamp'].dt.year
        daily_data['month'] = daily_data['timestamp'].dt.month
        daily_data['day'] = daily_data['timestamp'].dt.day
        daily_data['day_of_week'] = daily_data['timestamp'].dt.dayofweek
        daily_data['is_weekend'] = daily_data['day_of_week'] >= 5
        daily_data['week_of_year'] = daily_data['timestamp'].dt.isocalendar().week

        # Get unique years and months
        years = sorted(daily_data['year'].unique())

        # Create figure with subplots for each month
        n_years = len(years)
        fig = plt.figure(figsize=(16, 4 * n_years))

        for year_idx, year in enumerate(years):
            year_data = daily_data[daily_data['year'] == year]
            months = sorted(year_data['month'].unique())

            for month_idx, month in enumerate(months):
                month_data = year_data[year_data['month'] == month]

                # Create subplot
                ax = plt.subplot(n_years, 12, year_idx * 12 + month)

                # Create calendar grid
                # Get first day of month and number of days
                first_day = pd.Timestamp(year=year, month=month, day=1)
                first_weekday = first_day.dayofweek

                # Determine number of days in month
                if month == 12:
                    last_day = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
                else:
                    last_day = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
                n_days = last_day.day

                # Create grid (7 columns for days of week, up to 6 rows for weeks)
                calendar_grid = np.full((6, 7), np.nan)

                # Fill grid with average speeds
                for _, row in month_data.iterrows():
                    day = row['day']
                    day_of_week = row['day_of_week']
                    speed = row['avg_speed']

                    # Calculate position in grid
                    week_offset = (day + first_weekday - 1) // 7
                    calendar_grid[week_offset, day_of_week] = speed

                # Get route color for colormap
                route_color = self._get_route_color(route_code)

                # Create custom colormap
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list(
                    'route_cmap',
                    ['#f7f7f7', route_color],
                    N=256
                )

                # Plot heatmap
                im = ax.imshow(calendar_grid, cmap=cmap, aspect='auto',
                              vmin=np.nanmin(daily_data['avg_speed']),
                              vmax=np.nanmax(daily_data['avg_speed']))

                # Add day numbers
                for day in range(1, n_days + 1):
                    day_of_week = (day + first_weekday - 1) % 7
                    week_offset = (day + first_weekday - 1) // 7

                    # Check if this day has data
                    day_data = month_data[month_data['day'] == day]
                    if not day_data.empty:
                        is_weekend = day_data['is_weekend'].iloc[0]
                        text_color = 'darkred' if is_weekend else 'black'
                        fontweight = 'bold' if is_weekend else 'normal'
                    else:
                        text_color = 'gray'
                        fontweight = 'normal'

                    ax.text(day_of_week, week_offset, str(day),
                           ha='center', va='center', fontsize=8,
                           color=text_color, fontweight=fontweight)

                # Format axes
                ax.set_xticks(range(7))
                ax.set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'], fontsize=8)
                ax.set_yticks([])
                ax.set_title(f'{first_day.strftime("%B %Y")}', fontsize=9)

                # Remove spines
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Add overall title
        fig.suptitle(f'Calendar Heatmap: {self._get_route_label(route_code)}\n'
                    f'Daily Average Speed (Weekends in Bold Red)',
                    fontsize=14, fontweight='bold', y=0.995)

        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Avg Speed (km/h)', fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.93, 0.99])
        plt.show()

    def plot_time_series_decomposition(self, route_code: str) -> None:
        """
        Generate time series decomposition plot separating trend, seasonal, and residual components.

        Uses statsmodels seasonal_decompose to decompose the time series into:
        - Original: Raw time series data
        - Trend: Long-term progression
        - Seasonal: Weekly repeating patterns (period = 168 hours)
        - Residual: Random fluctuations after removing trend and seasonality

        Requires at least 2 weeks (336 hours) of data for reliable weekly seasonality detection.

        Parameters
        ----------
        route_code : str
            Route identifier

        Raises
        ------
        ValueError
            If no data found for the specified route

        Warnings
        --------
        Issues warning if insufficient data (< 2 weeks) for reliable decomposition

        Examples
        --------
        >>> viz.plot_time_series_decomposition('VJRQ+2M|RMJJ+F4')
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            raise ValueError(f"No data found for route: {route_code}")

        # Sort by timestamp and set as index
        route_data = route_data.sort_values('timestamp')
        route_data = route_data.set_index('timestamp')

        # Check for sufficient data (need at least 2 weeks for weekly seasonality)
        min_required_hours = 2 * 24 * 7  # 2 weeks = 336 hours
        n_observations = len(route_data)

        if n_observations < min_required_hours:
            warnings.warn(
                f"Insufficient data for reliable decomposition. "
                f"Need at least {min_required_hours} hours (2 weeks), "
                f"but only have {n_observations} hours. "
                f"Results may be unreliable."
            )
            # Still proceed but with a warning
            if n_observations < 24 * 7:  # Less than 1 week
                print(f"Error: Cannot perform decomposition with less than 1 week of data.")
                return

        # Resample to hourly frequency to handle any gaps
        # Use forward fill for small gaps (up to 3 hours)
        route_data_hourly = route_data['avg_speed'].resample('h').mean()
        route_data_hourly = route_data_hourly.ffill(limit=3)

        # If still have NaN values, use backward fill
        route_data_hourly = route_data_hourly.bfill(limit=3)

        # Drop any remaining NaN values
        route_data_hourly = route_data_hourly.dropna()

        if len(route_data_hourly) < min_required_hours:
            warnings.warn(
                f"After handling missing data, only {len(route_data_hourly)} hours remain. "
                f"Decomposition may be unreliable."
            )

        try:
            # Perform seasonal decomposition
            # Period = 168 hours (24 hours/day * 7 days/week) for weekly seasonality
            decomposition = seasonal_decompose(
                route_data_hourly,
                model='additive',
                period=24 * 7,  # Weekly seasonality
                extrapolate_trend='freq'
            )

            # Get route color
            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code)

            # Create figure with 4 subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))

            # Plot 1: Original time series
            axes[0].plot(decomposition.observed.index, decomposition.observed.values,
                        color=route_color, linewidth=1, alpha=0.8)
            axes[0].set_ylabel('Speed (km/h)', fontsize=10)
            axes[0].set_title(f'Time Series Decomposition: {route_label}',
                            fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].text(0.02, 0.95, 'Original', transform=axes[0].transAxes,
                        fontsize=10, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 2: Trend component
            axes[1].plot(decomposition.trend.index, decomposition.trend.values,
                        color=route_color, linewidth=2)
            axes[1].set_ylabel('Speed (km/h)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].text(0.02, 0.95, 'Trend', transform=axes[1].transAxes,
                        fontsize=10, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 3: Seasonal component (weekly pattern)
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                        color=route_color, linewidth=1, alpha=0.8)
            axes[2].set_ylabel('Speed (km/h)', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].text(0.02, 0.95, 'Seasonal (Weekly)', transform=axes[2].transAxes,
                        fontsize=10, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 4: Residual component
            axes[3].plot(decomposition.resid.index, decomposition.resid.values,
                        color='gray', linewidth=0.5, alpha=0.6)
            axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[3].set_ylabel('Speed (km/h)', fontsize=10)
            axes[3].set_xlabel('Date', fontsize=10)
            axes[3].grid(True, alpha=0.3)
            axes[3].text(0.02, 0.95, 'Residual', transform=axes[3].transAxes,
                        fontsize=10, fontweight='bold', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Format x-axis for all subplots
            for ax in axes:
                ax.tick_params(axis='x', rotation=45)
                # Format dates nicely
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.tight_layout()
            plt.show()

        except Exception as e:
            warnings.warn(f"Decomposition failed: {e}. Unable to generate plot.")
            print(f"Error details: {str(e)}")
            return

    def plot_hour_of_day_profiles(self, route_codes: list[str] | None = None) -> None:
        """
        Generate hour-of-day profile plots comparing selected routes on the same axes.

        Creates a line plot showing average speed by hour of day for all routes:
        - X-axis: Hour of day (0-23)
        - Y-axis: Average speed (km/h)
        - Each route is shown as a line with its designated color
        - Confidence bands (±1 standard deviation) are shown as shaded regions
        - Legend identifies each route

        This visualization enables easy comparison of route performance patterns
        throughout the day, revealing peak congestion hours and optimal travel times.

        Examples
        --------
        >>> viz.plot_hour_of_day_profiles()
        >>> viz.plot_hour_of_day_profiles(['ROUTE_A', 'ROUTE_B'])
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        selected_routes = self.routes if route_codes is None else [r for r in self.routes if r in set(route_codes)]

        if not selected_routes:
            warnings.warn("No valid routes selected for hour-of-day profile plot.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Compute hourly statistics for each route
        for route_code in selected_routes:
            route_data = df[df['route_code'] == route_code]

            if route_data.empty:
                continue

            # Group by hour and compute mean and std
            hourly_stats = route_data.groupby('hour')['avg_speed'].agg(['mean', 'std', 'count']).reset_index()

            # Handle missing hours (fill with NaN)
            hourly_stats = hourly_stats.set_index('hour').reindex(range(24)).reset_index()

            # Get route color and label
            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code)

            # Plot mean line
            ax.plot(hourly_stats['hour'], hourly_stats['mean'],
                   color=route_color, linewidth=2, label=route_label, alpha=0.9)

            # Add confidence band (±1 standard deviation)
            # Only show confidence band where we have data
            valid_mask = hourly_stats['mean'].notna()
            if valid_mask.any():
                hours = hourly_stats.loc[valid_mask, 'hour']
                means = hourly_stats.loc[valid_mask, 'mean']
                stds = hourly_stats.loc[valid_mask, 'std'].fillna(0)

                lower_bound = means - stds
                upper_bound = means + stds

                ax.fill_between(hours, lower_bound, upper_bound,
                               color=route_color, alpha=0.15, linewidth=0)

        # Format plot
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title('Hour-of-Day Speed Profiles\n'
                    '(Shaded regions show ±1 standard deviation)',
                    fontsize=14, fontweight='bold', pad=20)

        # Set x-axis ticks and labels
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([self._format_hour_label(h) for h in range(0, 24, 2)],
                          rotation=45, ha='right')
        ax.set_xlim(0, 23)
        ax.margins(x=0)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add legend
        if selected_routes:
            ax.legend(loc='best', framealpha=0.9, fontsize=9)

        # Set reasonable y-axis limits
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """
        Generate correlation matrix showing routes with similar temporal patterns.

        Computes correlation between routes' hourly speed patterns to identify
        routes that experience similar traffic conditions. High correlation indicates
        routes that tend to be fast or slow at the same times.

        The correlation is computed using Pearson correlation on hourly average speeds
        across all time periods. Routes with similar temporal patterns will have
        correlation coefficients close to 1.

        Uses seaborn heatmap with:
        - Diverging colormap (red for positive correlation, blue for negative)
        - Annotations showing correlation coefficients
        - Route labels for easy identification

        Examples
        --------
        >>> viz.plot_correlation_matrix()
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Create pivot table: rows = timestamps, columns = routes, values = avg_speed
        # First, create a unique timestamp identifier for each observation
        df_copy = df.copy()
        df_copy['timestamp_key'] = df_copy['timestamp'].astype(str)

        # Pivot to get routes as columns
        pivot_data = df_copy.pivot_table(
            index='timestamp_key',
            columns='route_code',
            values='avg_speed',
            aggfunc='mean'
        )

        # Compute correlation matrix
        correlation_matrix = pivot_data.corr(method='pearson')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            ax=ax
        )

        # Format labels - use short route labels
        route_labels = [self._get_route_label(route, 'short') for route in correlation_matrix.columns]
        ax.set_xticklabels(route_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(route_labels, rotation=0, fontsize=9)

        # Set title
        ax.set_title('Route Temporal Pattern Correlation Matrix\n'
                    '(Pearson correlation of hourly speed patterns)',
                    fontsize=12, fontweight='bold', pad=15)

        plt.tight_layout()
        plt.show()

    def create_ranking_animation(self) -> None:
        """
        Create animation showing how route rankings change throughout the day.

        Generates a series of bar charts showing route rankings for each hour of the day
        (0-23). Each frame shows the average speed ranking for that hour across all days
        in the dataset.

        The animation reveals:
        - Which routes are fastest/slowest at different times
        - How rankings shift during rush hours vs off-peak times
        - Temporal patterns in route performance

        Uses matplotlib animation to create an animated bar chart where:
        - X-axis: Routes (ordered by ranking for that hour)
        - Y-axis: Average speed (km/h)
        - Bars are colored according to route color palette
        - Title shows current hour

        Note: This creates a series of static plots showing each hour. For a true
        animation, use matplotlib.animation.FuncAnimation in an interactive environment.

        Examples
        --------
        >>> viz.create_ranking_animation()
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Compute average speed by route and hour
        hourly_rankings = df.groupby(['hour', 'route_code'])['avg_speed'].mean().reset_index()

        # Create figure for animation frames
        fig, ax = plt.subplots(figsize=(12, 7))

        # Store all frames
        frames_data = []

        for hour in range(24):
            # Get data for this hour
            hour_data = hourly_rankings[hourly_rankings['hour'] == hour].copy()

            if hour_data.empty:
                continue

            # Sort by average speed (descending) to get rankings
            hour_data = hour_data.sort_values('avg_speed', ascending=False).reset_index(drop=True)
            hour_data['rank'] = range(1, len(hour_data) + 1)

            frames_data.append(hour_data)

        # Determine if we should create animation or static plots
        # For now, create a grid of static plots showing key hours
        key_hours = [0, 6, 9, 12, 15, 18, 21, 23]  # Midnight, morning, midday, evening, night

        n_hours = len(key_hours)
        n_cols = 4
        n_rows = (n_hours + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_hours > 1 else [axes]

        for idx, hour in enumerate(key_hours):
            ax = axes[idx]

            # Get data for this hour
            hour_data = hourly_rankings[hourly_rankings['hour'] == hour].copy()

            if hour_data.empty:
                ax.text(0.5, 0.5, f'No data for {self._format_hour_label(hour)}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Sort by average speed (descending)
            hour_data = hour_data.sort_values('avg_speed', ascending=False).reset_index(drop=True)

            # Get colors for each route
            colors = [self._get_route_color(route) for route in hour_data['route_code']]

            # Create bar chart
            bars = ax.bar(range(len(hour_data)), hour_data['avg_speed'], color=colors, alpha=0.8)

            # Add route labels
            route_labels = [self._get_route_label(route, 'short') for route in hour_data['route_code']]
            ax.set_xticks(range(len(hour_data)))
            ax.set_xticklabels(route_labels, rotation=45, ha='right', fontsize=9)

            # Format axes
            ax.set_ylabel('Avg Speed (km/h)', fontsize=10)
            ax.set_title(f'{self._format_hour_label(hour)}', fontsize=11, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

            # Set consistent y-axis limits across all subplots
            ax.set_ylim(0, hourly_rankings['avg_speed'].max() * 1.1)

            # Add ranking numbers on bars
            for i, (bar, speed) in enumerate(zip(bars, hour_data['avg_speed'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                       f'#{i+1}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Hide extra subplots
        for idx in range(n_hours, len(axes)):
            axes[idx].set_visible(False)

        # Add overall title
        fig.suptitle('Route Rankings Throughout the Day\n'
                    '(Average speed rankings by hour - higher is better)',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

        # Print summary statistics
        print("\nRanking Volatility Analysis:")
        print("=" * 60)

        # Calculate how much each route's ranking changes throughout the day
        for route in self.routes:
            route_rankings = []
            for hour_data in frames_data:
                route_hour = hour_data[hour_data['route_code'] == route]
                if not route_hour.empty:
                    route_rankings.append(route_hour['rank'].iloc[0])

            if route_rankings:
                avg_rank = np.mean(route_rankings)
                std_rank = np.std(route_rankings)
                min_rank = np.min(route_rankings)
                max_rank = np.max(route_rankings)
                rank_range = max_rank - min_rank

                route_label = self._get_route_label(route, 'short')
                print(f"{route_label:20s} | Avg Rank: {avg_rank:.1f} | "
                      f"Std: {std_rank:.1f} | Range: {min_rank}-{max_rank} ({rank_range})")

        print("=" * 60)
        print("Note: Higher rank range indicates more volatile performance throughout the day")

    def plot_parallel_coordinates(self, metrics: Optional[List[str]] = None, route_codes: Optional[List[str]] = None) -> None:
        """
        Generate parallel coordinates plot showing multiple metrics simultaneously.

        This visualization displays multiple metrics for each route as parallel vertical axes,
        with lines connecting the values for each route. This allows comparison of routes
        across multiple dimensions at once.

        Parameters
        ----------
        metrics : Optional[List[str]], default None
            List of metric names to display. If None, uses default metrics:
            ['avg_speed', 'duration', 'distance', 'speed_variance']
            Available metrics: 'avg_speed', 'duration', 'distance', 'speed_variance',
            'speed_std', 'speed_min', 'speed_max'
        route_codes : Optional[List[str]], default None
            Route codes to include in the plot. If None, plots all routes.

        Returns
        -------
        None
            Displays the plot using matplotlib

        Raises
        ------
        ValueError
            If specified metrics are not available in the dataset

        Notes
        -----
        - Each route is represented by a line connecting its values across all metrics
        - Metrics are normalized to [0, 1] scale for visual comparison
        - Route colors are consistent with the color palette from routes_df
        - Higher values are better for speed metrics, lower for duration

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> viz.plot_parallel_coordinates(['avg_speed', 'duration', 'speed_variance'])
        """
        # Default metrics if none specified
        if metrics is None:
            metrics = ['avg_speed', 'speed_min', 'speed_max', 'speed_std', 'speed_variance', 'duration', 'distance']

        # Filter to selected routes when provided
        df = self.df
        if route_codes is not None:
            route_codes = [r for r in self.routes if r in set(route_codes)]
            df = df[df['route_code'].isin(route_codes)].copy()

        if df.empty:
            # Create an empty chart scaffold
            self._set_figure_style(figsize=(14, 8))
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.set_title('No valid routes selected for parallel coordinates plot.',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Normalized Value (0-1 scale)')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            plt.tight_layout()
            plt.show()
            return

        # Compute route-level aggregated metrics
        route_metrics = df.groupby('route_code').agg({
            'avg_speed': ['mean', 'std', 'min', 'max'],
            'duration': 'mean',
            'distance': 'mean'
        }).reset_index()

        # Flatten column names
        route_metrics.columns = ['route_code', 'avg_speed', 'speed_std', 'speed_min',
                                  'speed_max', 'duration', 'distance']

        # Add variance metric
        route_metrics['speed_variance'] = df.groupby('route_code')['avg_speed'].var().values

        # Validate requested metrics
        available_metrics = ['avg_speed', 'duration', 'distance', 'speed_variance',
                            'speed_std', 'speed_min', 'speed_max']
        invalid_metrics = set(metrics) - set(available_metrics)
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. "
                           f"Available metrics: {available_metrics}")

        # Select only requested metrics
        plot_data = route_metrics[['route_code'] + metrics].copy()

        # Normalize metrics to [0, 1] scale for visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        plot_data[metrics] = scaler.fit_transform(plot_data[metrics])

        # Create figure
        self._set_figure_style(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up x-axis positions for each metric
        x_positions = np.arange(len(metrics))

        # Plot each route as a line
        for _, row in plot_data.iterrows():
            route_code = row['route_code']
            values = row[metrics].values
            color = self._get_route_color(route_code)
            label = self._get_route_label(route_code, label_type='short')

            ax.plot(x_positions, values, marker='o', linewidth=2,
                   color=color, label=label, alpha=0.7)

        # Customize axes
        ax.set_xticks(x_positions)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                          rotation=15, ha='right')
        ax.set_ylabel('Normalized Value (0-1 scale)', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Multi-Dimensional Route Comparison\n(Parallel Coordinates)',
                    fontsize=14, fontweight='bold', pad=20)

        # Add grid for readability
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                 frameon=True, fancybox=True, shadow=True)

        # Add note about normalization
        fig.text(0.5, 0.02,
                'Note: All metrics normalized to 0-1 scale for comparison. '
                'Higher values indicate better performance for speed metrics, '
                'lower values for duration.',
                ha='center', fontsize=9, style='italic', color='gray')

        plt.tight_layout()
        plt.show()

    def plot_radar_chart(self, routes: Optional[List[str]] = None) -> None:
        """
        Generate radar/spider chart comparing routes across multiple dimensions.

        This visualization displays routes as polygons on a circular plot, with each
        axis representing a different performance dimension. This allows intuitive
        visual comparison of route strengths and weaknesses.

        Parameters
        ----------
        routes : Optional[List[str]], default None
            List of route codes to compare. If None, includes all routes.
            Maximum 6 routes recommended for readability.

        Returns
        -------
        None
            Displays the plot using matplotlib

        Raises
        ------
        ValueError
            If specified routes are not found in the dataset

        Notes
        -----
        - Dimensions include: mean speed, speed consistency (inverse variance),
          peak-hour performance, off-peak performance, weekend performance
        - All dimensions are normalized to [0, 1] scale
        - Larger polygons indicate better overall performance
        - Route colors are consistent with the color palette from routes_df

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> viz.plot_radar_chart(['VJRQ+2M|RMJJ+F4', 'XMW9+G8|WMJR+V4'])
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Select routes to plot
        auto_selected_routes = routes is None
        if routes is None:
            routes = sorted(df['route_code'].unique())
        else:
            # Validate routes
            invalid_routes = set(routes) - set(df['route_code'].unique())
            if invalid_routes:
                raise ValueError(f"Routes not found in dataset: {invalid_routes}")

        # Limit to 6 routes for readability
        if len(routes) > 6:
            if not auto_selected_routes:
                warnings.warn(f"Too many routes ({len(routes)}). Showing first 6 for readability.")
            routes = routes[:6]

        # Define dimensions to compare
        dimensions = []
        dimension_labels = []

        # 1. Mean Speed
        mean_speeds = df.groupby('route_code')['avg_speed'].mean()
        dimensions.append(mean_speeds)
        dimension_labels.append('Mean Speed')

        # 2. Speed Consistency (inverse of coefficient of variation)
        speed_cv = df.groupby('route_code')['avg_speed'].std() / df.groupby('route_code')['avg_speed'].mean()
        consistency = 1 / (1 + speed_cv)  # Transform so higher is better
        dimensions.append(consistency)
        dimension_labels.append('Consistency')

        # 3. Peak-Hour Performance (morning rush: 7-10am, evening rush: 5-8pm)
        peak_hours = df[df['hour'].isin(list(range(7, 10)) + list(range(17, 20)))]
        peak_speed = peak_hours.groupby('route_code')['avg_speed'].mean()
        dimensions.append(peak_speed)
        dimension_labels.append('Peak-Hour\nPerformance')

        # 4. Off-Peak Performance (10am-5pm)
        offpeak_hours = df[df['hour'].isin(range(10, 17))]
        offpeak_speed = offpeak_hours.groupby('route_code')['avg_speed'].mean()
        dimensions.append(offpeak_speed)
        dimension_labels.append('Off-Peak\nPerformance')

        # 5. Weekend Performance
        weekend_data = df[df['is_weekend'] == True]
        if len(weekend_data) > 0:
            weekend_speed = weekend_data.groupby('route_code')['avg_speed'].mean()
            dimensions.append(weekend_speed)
            dimension_labels.append('Weekend\nPerformance')

        # Combine into DataFrame
        radar_data = pd.DataFrame(dimensions).T
        radar_data.columns = dimension_labels

        # Normalize to [0, 1] scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        radar_data_normalized = pd.DataFrame(
            scaler.fit_transform(radar_data),
            index=radar_data.index,
            columns=radar_data.columns
        )

        # Filter to selected routes
        radar_data_normalized = radar_data_normalized.loc[routes]

        # Set up radar chart
        num_vars = len(dimension_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot each route
        for route_code in routes:
            values = radar_data_normalized.loc[route_code].values.tolist()
            values += values[:1]  # Complete the circle

            color = self._get_route_color(route_code)
            label = self._get_route_label(route_code, label_type='short')

            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add title
        ax.set_title('Route Performance Comparison\n(Radar Chart)',
                    fontsize=14, fontweight='bold', pad=30)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                 frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.show()

    def plot_speed_duration_scatter(self) -> None:
        """
        Generate scatter plot showing speed vs duration relationship with marginal distributions.

        This visualization displays the relationship between average speed and duration
        for each route, with histograms along the margins showing the distribution of
        each variable. This helps identify routes with consistent vs variable performance.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Displays the plot using matplotlib

        Notes
        -----
        - Main scatter plot shows speed vs duration for all observations
        - Top marginal histogram shows speed distribution
        - Right marginal histogram shows duration distribution
        - Each route is colored according to the color palette from routes_df
        - Helps identify routes with high speed but high variability

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> viz.plot_speed_duration_scatter()
        """
        # Create figure with gridspec for marginal plots
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                             height_ratios=[1, 3, 0.2], width_ratios=[3, 1, 0.2])

        # Main scatter plot
        ax_main = fig.add_subplot(gs[1, 0])

        # Marginal histograms
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        # Plot scatter for each route
        routes = sorted(self.df['route_code'].unique())

        for route_code in routes:
            route_data = self.df[self.df['route_code'] == route_code]
            color = self._get_route_color(route_code)
            label = self._get_route_label(route_code, label_type='short')

            # Main scatter
            ax_main.scatter(route_data['avg_speed'], route_data['duration'],
                          alpha=0.4, s=20, color=color, label=label, edgecolors='none')

            # Top histogram (speed distribution)
            ax_top.hist(route_data['avg_speed'], bins=30, alpha=0.5,
                       color=color, edgecolor='none')

            # Right histogram (duration distribution)
            ax_right.hist(route_data['duration'], bins=30, alpha=0.5,
                         color=color, orientation='horizontal', edgecolor='none')

        # Customize main scatter plot
        ax_main.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Duration (minutes)', fontsize=12, fontweight='bold')
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.set_axisbelow(True)

        # Add legend to main plot
        ax_main.legend(loc='upper right', frameon=True, fancybox=True,
                      shadow=True, fontsize=9)

        # Customize marginal histograms
        ax_top.set_ylabel('Frequency', fontsize=10)
        ax_top.tick_params(labelbottom=False)
        ax_top.grid(True, alpha=0.3, linestyle='--', axis='y')

        ax_right.set_xlabel('Frequency', fontsize=10)
        ax_right.tick_params(labelleft=False)
        ax_right.grid(True, alpha=0.3, linestyle='--', axis='x')

        # Add title
        fig.suptitle('Speed vs Duration Relationship with Marginal Distributions',
                    fontsize=14, fontweight='bold', y=0.98)

        # Add interpretation note
        fig.text(0.5, 0.02,
                'Note: Scatter plot shows individual observations. '
                'Marginal histograms show distribution of speed (top) and duration (right). '
                'Tighter clusters indicate more consistent route performance.',
                ha='center', fontsize=9, style='italic', color='gray', wrap=True)

        plt.show()

    def plot_time_of_day_facets(self) -> None:
        """
        Generate interactive single-view visualization of speed distributions by time-of-day.

        Displays one time-of-day category at a time with Previous/Next buttons to navigate.
        Each chart is rendered at 300 DPI for high clarity.

        Time categories (matching data_utils.compute_temporal_features):
        - Late Night: 12-6 AM
        - Morning: 6-8 AM
        - Morning Rush: 8-11 AM
        - Early Afternoon: 11 AM - 2 PM
        - Late Afternoon: 2-6 PM
        - Evening Rush: 6-9 PM
        - Night: 9 PM - 12 AM

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> viz.plot_time_of_day_facets()
        """
        from IPython.display import display, clear_output
        import ipywidgets as widgets

        df_with_features = self._ensure_temporal_features(self.df)

        time_categories = [
            'late_night', 'morning', 'morning_rush',
            'early_afternoon', 'late_afternoon',
            'evening_rush', 'night',
        ]
        time_labels = {
            'late_night': 'Late Night (12-6 AM)',
            'morning': 'Morning (6-8 AM)',
            'morning_rush': 'Morning Rush (8-11 AM)',
            'early_afternoon': 'Early Afternoon (11 AM - 2 PM)',
            'late_afternoon': 'Late Afternoon (2-6 PM)',
            'evening_rush': 'Evening Rush (6-9 PM)',
            'night': 'Night (9 PM - 12 AM)',
        }

        routes = sorted(df_with_features['route_code'].unique())

        def render_chart(idx: int):
            time_cat = time_categories[idx]
            time_data = df_with_features[
                df_with_features['time_category'] == time_cat
            ]

            fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

            if time_data.empty:
                ax.text(0.5, 0.5, f'No data for {time_labels[time_cat]}',
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(time_labels[time_cat], fontsize=14, fontweight='bold', pad=15)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                data_by_route = []
                colors = []
                labels = []

                for route_code in routes:
                    route_data = time_data[time_data['route_code'] == route_code]
                    if not route_data.empty:
                        data_by_route.append(route_data['avg_speed'].values)
                        colors.append(self._get_route_color(route_code))
                        labels.append(self._get_route_label(route_code, label_type='short'))

                if data_by_route:
                    parts = ax.violinplot(
                        data_by_route, positions=range(len(data_by_route)),
                        showmeans=True, showmedians=True, widths=0.7,
                    )

                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors[i])
                        pc.set_alpha(0.7)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(1)

                    parts['cmeans'].set_edgecolor('darkred')
                    parts['cmeans'].set_linewidth(2)
                    parts['cmedians'].set_edgecolor('darkblue')
                    parts['cmedians'].set_linewidth(2)

                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
                    ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
                    ax.set_axisbelow(True)

                    total_samples = len(time_data)
                    ax.text(
                        0.98, 0.98, f'n={total_samples:,}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=10, style='italic', color='gray',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='gray', alpha=0.7),
                    )

            ax.set_title(
                f'{time_labels[time_cat]}  ({idx + 1}/{len(time_categories)})',
                fontsize=14, fontweight='bold', pad=15,
            )
            fig.text(
                0.5, 0.01,
                'Red line = mean, Blue line = median. Wider sections = more common speeds.',
                ha='center', fontsize=10, style='italic', color='gray',
            )
            plt.tight_layout(rect=[0, 0.03, 1, 1])
            plt.show()

        state = {'idx': 0}
        out = widgets.Output()

        def refresh():
            with out:
                clear_output(wait=True)
                render_chart(state['idx'])

        def on_prev(_):
            state['idx'] = (state['idx'] - 1) % len(time_categories)
            refresh()

        def on_next(_):
            state['idx'] = (state['idx'] + 1) % len(time_categories)
            refresh()

        prev_btn = widgets.Button(description='◀ Previous', button_style='info')
        next_btn = widgets.Button(description='Next ▶', button_style='info')
        prev_btn.on_click(on_prev)
        next_btn.on_click(on_next)

        display(widgets.VBox([widgets.HBox([prev_btn, next_btn]), out]))
        refresh()







    # ========================================================================
    # CDF and Ranking Evolution Visualizations
    # ========================================================================

    def plot_cdf_comparison(self) -> None:
        """
        Generate cumulative distribution function (CDF) comparison for travel time reliability.

        Creates CDF plots comparing the distribution of travel times (or speeds) across routes.
        The CDF shows the probability that a route's speed is less than or equal to a given value.

        Key insights from CDF:
        - Steeper curves indicate more consistent performance (less variability)
        - Curves shifted to the right indicate faster routes
        - The median (50th percentile) shows typical performance
        - The spread shows reliability (narrow = reliable, wide = variable)

        Uses route-specific colors from the color palette for consistency.

        Examples
        --------
        >>> viz.plot_cdf_comparison()
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot CDF for each route
        for route_code in self.routes:
            route_data = df[df['route_code'] == route_code]

            if route_data.empty:
                continue

            # Get speeds and sort
            speeds = route_data['avg_speed'].dropna().sort_values()

            if len(speeds) == 0:
                continue

            # Compute CDF (empirical cumulative distribution)
            cdf = np.arange(1, len(speeds) + 1) / len(speeds)

            # Get route color and label
            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code)

            # Plot CDF
            ax.plot(speeds, cdf, color=route_color, linewidth=2.5,
                   label=route_label, alpha=0.85)

            # Add markers at key percentiles (25th, 50th, 75th)
            percentiles = [0.25, 0.50, 0.75]
            for p in percentiles:
                idx = int(p * len(speeds))
                if idx < len(speeds):
                    ax.plot(speeds.iloc[idx], p, 'o', color=route_color,
                           markersize=6, markeredgecolor='white', markeredgewidth=1.5)

        # Format plot
        ax.set_xlabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Distribution Function (CDF) Comparison\n'
                    'Travel Time Reliability Analysis',
                    fontsize=14, fontweight='bold', pad=20)

        # Set y-axis to percentage
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.1)])

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add reference lines for key percentiles
        for p, label in [(0.25, '25th'), (0.50, '50th (Median)'), (0.75, '75th')]:
            ax.axhline(y=p, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.text(ax.get_xlim()[0], p, f'  {label}', va='center',
                   fontsize=9, color='gray', style='italic')

        # Add legend
        ax.legend(loc='lower right', framealpha=0.95, fontsize=10)

        # Add interpretation note
        fig.text(0.5, 0.01,
                'Interpretation: Steeper curves = more consistent performance. '
                'Curves to the right = faster routes. '
                'Markers show 25th, 50th (median), and 75th percentiles.',
                ha='center', fontsize=10, style='italic', color='gray', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    def plot_ranking_evolution(self, days_rolling: int = 10) -> None:
        """
        Show how route rankings change over time using a rolling window.

        Creates a line plot showing each route's ranking position over time.
        Rankings are computed using a rolling window (default 10 days) to smooth
        out daily fluctuations and reveal longer-term trends.

        Key insights:
        - Stable routes maintain consistent ranking positions
        - Volatile routes show frequent ranking changes
        - Crossing lines indicate shifts in relative performance
        - Trends reveal improving or declining route performance

        Parameters
        ----------
        days_rolling : int, default=10
            Number of days in the rolling window for ranking calculation

        Examples
        --------
        >>> viz.plot_ranking_evolution(days_rolling=14)
        """
        from traffic_analyzer import TrafficAnalyzer

        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Get date range
        dates = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='D'
        )

        # Initialize analyzer
        analyzer = TrafficAnalyzer(self.df, self.routes_df)

        # Compute rankings for each date
        rankings_over_time = []

        for date in dates:
            try:
                # Calculate R³S² scores for this date
                scores = analyzer.calculate_rrs(ref_date=date, days_rolling=days_rolling)

                # Add date and ranking
                scores['date'] = date
                scores['rank'] = range(1, len(scores) + 1)

                rankings_over_time.append(scores[['date', 'route_code', 'rank', 'points']])
            except Exception as e:
                # Skip dates with insufficient data
                continue

        if not rankings_over_time:
            print("Insufficient data to compute ranking evolution.")
            return

        # Combine all rankings
        rankings_df = pd.concat(rankings_over_time, ignore_index=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot ranking evolution for each route
        for route_code in self.routes:
            route_rankings = rankings_df[rankings_df['route_code'] == route_code]

            if route_rankings.empty:
                continue

            # Get route color and label
            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code)

            # Plot ranking over time
            ax.plot(route_rankings['date'], route_rankings['rank'],
                   color=route_color, linewidth=2, label=route_label,
                   marker='o', markersize=3, alpha=0.8)

        # Format plot
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ranking Position', fontsize=12, fontweight='bold')
        ax.set_title(f'Route Ranking Evolution Over Time\n'
                    f'({days_rolling}-day rolling window)',
                    fontsize=14, fontweight='bold', pad=20)

        # Invert y-axis so rank 1 is at the top
        ax.invert_yaxis()

        # Set y-axis ticks to integer ranks
        n_routes = len(self.routes)
        ax.set_yticks(range(1, n_routes + 1))
        ax.set_yticklabels([f'#{i}' for i in range(1, n_routes + 1)])

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Format x-axis dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=10, ncol=2)

        # Add interpretation note
        fig.text(0.5, 0.01,
                f'Interpretation: Lower rank numbers (#1) indicate better performance. '
                f'Stable lines = consistent performance. Crossing lines = changing relative performance.',
                ha='center', fontsize=10, style='italic', color='gray', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

        # Print ranking volatility summary
        print("\nRanking Volatility Summary:")
        print("=" * 70)
        for route_code in self.routes:
            route_rankings = rankings_df[rankings_df['route_code'] == route_code]
            if not route_rankings.empty:
                avg_rank = route_rankings['rank'].mean()
                std_rank = route_rankings['rank'].std()
                min_rank = route_rankings['rank'].min()
                max_rank = route_rankings['rank'].max()
                rank_range = max_rank - min_rank

                route_label = self._get_route_label(route_code, 'short')
                print(f"{route_label:25s} | Avg: #{avg_rank:.1f} | "
                      f"Std: {std_rank:.2f} | Range: #{int(min_rank)}-#{int(max_rank)} ({int(rank_range)})")
        print("=" * 70)

    # ========================================================================
    # Anomaly Detection Visualizations
    # ========================================================================

    def plot_control_chart(self, route_code: str) -> None:
        """
        Generate control chart with 2-sigma and 3-sigma bounds for anomaly detection.

        Creates a control chart (also known as Shewhart chart) showing:
        - Time series of average speeds
        - Mean line (center line)
        - ±2 sigma bounds (warning limits)
        - ±3 sigma bounds (control limits)
        - Highlighted points exceeding 3 standard deviations

        Control charts help identify:
        - Unusual observations (outliers)
        - Systematic shifts in performance
        - Increasing/decreasing variability
        - Process stability over time

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> viz.plot_control_chart('VJRQ+2M|RMJJ+F4')
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Sort by timestamp
        route_data = route_data.sort_values('timestamp')

        # Compute statistics
        mean_speed = route_data['avg_speed'].mean()
        std_speed = route_data['avg_speed'].std()

        # Compute control limits
        ucl_3sigma = mean_speed + 3 * std_speed  # Upper control limit
        lcl_3sigma = mean_speed - 3 * std_speed  # Lower control limit
        uwl_2sigma = mean_speed + 2 * std_speed  # Upper warning limit
        lwl_2sigma = mean_speed - 2 * std_speed  # Lower warning limit

        # Identify anomalies (beyond 3 sigma)
        anomalies = route_data[
            (route_data['avg_speed'] > ucl_3sigma) |
            (route_data['avg_speed'] < lcl_3sigma)
        ]

        # Get route color and label
        route_color = self._get_route_color(route_code)
        route_label = self._get_route_label(route_code)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot time series
        ax.plot(route_data['timestamp'], route_data['avg_speed'],
               color=route_color, linewidth=1, alpha=0.6, label='Observed Speed')

        # Plot mean line
        ax.axhline(y=mean_speed, color='green', linestyle='-', linewidth=2,
                  label=f'Mean ({mean_speed:.1f} km/h)', alpha=0.8)

        # Plot 2-sigma bounds (warning limits)
        ax.axhline(y=uwl_2sigma, color='orange', linestyle='--', linewidth=1.5,
                  label='±2σ (Warning Limits)', alpha=0.7)
        ax.axhline(y=lwl_2sigma, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

        # Plot 3-sigma bounds (control limits)
        ax.axhline(y=ucl_3sigma, color='red', linestyle='--', linewidth=2,
                  label='±3σ (Control Limits)', alpha=0.8)
        ax.axhline(y=lcl_3sigma, color='red', linestyle='--', linewidth=2, alpha=0.8)

        # Highlight anomalies
        if not anomalies.empty:
            ax.scatter(anomalies['timestamp'], anomalies['avg_speed'],
                      color='red', s=100, marker='o', edgecolors='darkred',
                      linewidths=2, zorder=5, label=f'Anomalies (n={len(anomalies)})')

        # Format plot
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Control Chart: {route_label}\n'
                    f'Anomaly Detection with Statistical Process Control',
                    fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Format x-axis dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        # Add statistics box
        stats_text = (
            f'Statistics:\n'
            f'Mean: {mean_speed:.2f} km/h\n'
            f'Std Dev: {std_speed:.2f} km/h\n'
            f'Anomalies: {len(anomalies)} ({len(anomalies)/len(route_data)*100:.1f}%)'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        plt.show()


    def plot_deviation_timeline(self, route_code: str) -> None:
        """
        Generate deviation timeline with shaded regions for unusual periods.

        Creates a timeline showing:
        - Deviation from expected speed over time
        - Shaded regions for periods exceeding 2σ and 3σ thresholds
        - Contextual anomaly detection based on hour-of-day and day-of-week
        - Moving average trend line

        This visualization helps identify:
        - Temporal patterns in anomalous behavior
        - Sustained periods of unusual performance
        - Whether deviations are getting better or worse over time

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> viz.plot_deviation_timeline('VJRQ+2M|RMJJ+F4')
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Sort by timestamp
        route_data = route_data.sort_values('timestamp')

        # Compute expected speed based on hour-of-day and day-of-week patterns
        expected_speeds = route_data.groupby(
            ['hour', 'day_of_week']
        )['avg_speed'].transform('mean')

        route_data['expected_speed'] = expected_speeds
        route_data['deviation'] = route_data['avg_speed'] - route_data['expected_speed']

        # Compute deviation statistics
        mean_deviation = route_data['deviation'].mean()
        std_deviation = route_data['deviation'].std()

        # Compute moving average (7-day window)
        route_data['deviation_ma'] = route_data['deviation'].rolling(
            window=min(168, len(route_data)),  # 7 days * 24 hours, or less if insufficient data
            center=True,
            min_periods=1
        ).mean()

        # Get route color and label
        route_color = self._get_route_color(route_code)
        route_label = self._get_route_label(route_code)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot deviation timeline
        ax.plot(route_data['timestamp'], route_data['deviation'],
               color=route_color, linewidth=1, alpha=0.4, label='Deviation')

        # Plot moving average
        ax.plot(route_data['timestamp'], route_data['deviation_ma'],
               color='darkblue', linewidth=2.5, alpha=0.8, label='7-Day Moving Average')

        # Plot mean line
        ax.axhline(y=mean_deviation, color='green', linestyle='-', linewidth=2,
                  label=f'Mean Deviation ({mean_deviation:.1f} km/h)', alpha=0.8)

        # Add shaded regions for 2σ and 3σ thresholds
        # Positive deviations (faster than expected)
        ax.axhspan(mean_deviation + 2*std_deviation, mean_deviation + 3*std_deviation,
                  alpha=0.15, color='orange', label='±2σ Warning Zone')
        ax.axhspan(mean_deviation + 3*std_deviation, ax.get_ylim()[1],
                  alpha=0.2, color='red', label='±3σ Anomaly Zone')

        # Negative deviations (slower than expected)
        ax.axhspan(mean_deviation - 3*std_deviation, mean_deviation - 2*std_deviation,
                  alpha=0.15, color='orange')
        ax.axhspan(ax.get_ylim()[0], mean_deviation - 3*std_deviation,
                  alpha=0.2, color='red')

        # Highlight anomalous periods (beyond 3σ)
        anomalies = route_data[abs(route_data['deviation']) > 3*std_deviation]
        if not anomalies.empty:
            ax.scatter(anomalies['timestamp'], anomalies['deviation'],
                      color='red', s=80, marker='o', edgecolors='darkred',
                      linewidths=2, zorder=5, label=f'Anomalies (n={len(anomalies)})')

        # Format plot
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deviation from Expected Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Deviation Timeline: {route_label}\n'
                    f'Contextual Anomaly Detection with Temporal Patterns',
                    fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Format x-axis dates
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=9, ncol=2)

        # Count anomalies
        anomaly_pct = len(anomalies) / len(route_data) * 100

        # Add statistics box
        stats_text = (
            f'Statistics:\n'
            f'Mean Deviation: {mean_deviation:.2f} km/h\n'
            f'Std Deviation: {std_deviation:.2f} km/h\n'
            f'Anomalies: {len(anomalies)} ({anomaly_pct:.1f}%)\n'
            f'Max Positive: +{route_data["deviation"].max():.1f} km/h\n'
            f'Max Negative: {route_data["deviation"].min():.1f} km/h'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        plt.show()

    def plot_anomaly_scatter(self, route_code: Optional[str] = None) -> None:
        """
        Generate anomaly scatter plot highlighting deviations from expected patterns.

        Creates a scatter plot showing:
        - Expected speed (based on hour-of-day and day-of-week patterns)
        - Actual observed speed
        - Deviations colored by magnitude
        - Contextual anomaly detection (considers time-of-day patterns)

        This visualization helps identify:
        - When actual speeds deviate significantly from expected patterns
        - Whether deviations are positive (faster) or negative (slower)
        - Patterns in anomalous behavior

        Parameters
        ----------
        route_code : str, optional
            Route identifier. If None, shows all routes.

        Examples
        --------
        >>> viz.plot_anomaly_scatter('VJRQ+2M|RMJJ+F4')
        >>> viz.plot_anomaly_scatter()  # All routes
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data if route specified
        if route_code:
            plot_data = df[df['route_code'] == route_code].copy()
            if plot_data.empty:
                print(f"No data available for route: {self._get_route_label(route_code)}")
                return
            title_suffix = self._get_route_label(route_code)
        else:
            plot_data = df.copy()
            title_suffix = "All Routes"

        # Compute expected speed based on hour-of-day and day-of-week patterns
        # Group by route, hour, and day_of_week to get expected patterns
        expected_speeds = plot_data.groupby(
            ['route_code', 'hour', 'day_of_week']
        )['avg_speed'].transform('mean')

        plot_data['expected_speed'] = expected_speeds
        plot_data['deviation'] = plot_data['avg_speed'] - plot_data['expected_speed']

        # Compute deviation magnitude (in standard deviations)
        std_dev = plot_data.groupby(['route_code', 'hour', 'day_of_week'])['avg_speed'].transform('std')
        plot_data['deviation_sigma'] = plot_data['deviation'] / std_dev.replace(0, 1)  # Avoid division by zero

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create color map based on deviation magnitude
        # Use diverging colormap: blue (slower than expected), white (as expected), red (faster than expected)
        scatter = ax.scatter(
            plot_data['expected_speed'],
            plot_data['avg_speed'],
            c=plot_data['deviation_sigma'],
            cmap='RdBu_r',  # Red for positive deviations, Blue for negative
            s=30,
            alpha=0.6,
            edgecolors='none',
            vmin=-3,
            vmax=3
        )

        # Add diagonal line (y=x) representing perfect match
        min_speed = min(plot_data['expected_speed'].min(), plot_data['avg_speed'].min())
        max_speed = max(plot_data['expected_speed'].max(), plot_data['avg_speed'].max())
        ax.plot([min_speed, max_speed], [min_speed, max_speed],
               'k--', linewidth=2, alpha=0.5, label='Expected = Actual')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deviation (σ)', fontsize=11, fontweight='bold')
        cbar.ax.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        cbar.ax.axhline(y=-2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        cbar.ax.axhline(y=3, color='red', linestyle='--', linewidth=2, alpha=0.8)
        cbar.ax.axhline(y=-3, color='red', linestyle='--', linewidth=2, alpha=0.8)

        # Format plot
        ax.set_xlabel('Expected Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Anomaly Scatter Plot: {title_suffix}\n'
                    f'Contextual Anomaly Detection (Hour-of-Day & Day-of-Week)',
                    fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        # Count anomalies (beyond 3 sigma)
        anomalies = plot_data[abs(plot_data['deviation_sigma']) > 3]
        anomaly_pct = len(anomalies) / len(plot_data) * 100

        # Add statistics box
        stats_text = (
            f'Statistics:\n'
            f'Total Observations: {len(plot_data):,}\n'
            f'Anomalies (>3σ): {len(anomalies):,} ({anomaly_pct:.1f}%)\n'
            f'Mean Deviation: {plot_data["deviation"].mean():.2f} km/h\n'
            f'Std Deviation: {plot_data["deviation"].std():.2f} km/h'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()
        plt.show()


    def generate_outlier_summary(self, top_n: int = 20,
                                 route_code: Optional[str] = None,
                                 threshold_sigma: float = 3.0) -> pd.DataFrame:
        """
        Generate a tabular summary of the most anomalous observations.

        Anomalies are computed contextually by comparing each observation against
        the mean speed for the same route, hour-of-day, and day-of-week.

        Parameters
        ----------
        top_n : int, default=20
            Maximum number of anomalous rows to return.
        route_code : str, optional
            Restrict analysis to a single route.
        threshold_sigma : float, default=3.0
            Minimum absolute z-score of deviation to consider an outlier.

        Returns
        -------
        pd.DataFrame
            Top anomalies sorted by absolute deviation_sigma (descending).
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Optional route filter
        if route_code is not None:
            out_df = df[df['route_code'] == route_code].copy()
            if out_df.empty:
                warnings.warn(f"No data available for route: {route_code}")
                return pd.DataFrame()
        else:
            out_df = df.copy()

        # Contextual expected speed by route/hour/day
        out_df['expected_speed'] = out_df.groupby(
            ['route_code', 'hour', 'day_of_week']
        )['avg_speed'].transform('mean')
        out_df['deviation'] = out_df['avg_speed'] - out_df['expected_speed']

        # Convert deviation to sigma units; avoid divide-by-zero
        context_std = out_df.groupby(['route_code', 'hour', 'day_of_week'])['avg_speed'].transform('std')
        context_std = context_std.replace(0, np.nan)
        out_df['deviation_sigma'] = out_df['deviation'] / context_std
        out_df['deviation_sigma'] = out_df['deviation_sigma'].replace([np.inf, -np.inf], np.nan)

        # Filter and rank anomalies
        anomalies = out_df[out_df['deviation_sigma'].abs() >= threshold_sigma].copy()
        if anomalies.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'route_code', 'route_label', 'hour', 'day_of_week',
                'avg_speed', 'expected_speed', 'deviation', 'deviation_sigma'
            ])

        anomalies['abs_deviation_sigma'] = anomalies['deviation_sigma'].abs()
        anomalies['route_label'] = anomalies['route_code'].apply(self._get_route_label)

        anomalies = anomalies.sort_values('abs_deviation_sigma', ascending=False)
        anomalies = anomalies.head(top_n)

        result_cols = [
            'timestamp', 'route_code', 'route_label', 'hour', 'day_of_week',
            'avg_speed', 'expected_speed', 'deviation', 'deviation_sigma'
        ]
        result = anomalies[result_cols].copy()

        # Keep display-friendly precision
        for col in ['avg_speed', 'expected_speed', 'deviation', 'deviation_sigma']:
            result[col] = result[col].round(3)

        return result.reset_index(drop=True)




    def plot_residual_analysis(self, route_code: str) -> None:
        """
        Generate residual analysis plot using seasonal decomposition.

        Creates diagnostic plots for residuals from seasonal decomposition:
        1. Residual time series
        2. Residual histogram with normal distribution overlay
        3. Q-Q plot for normality assessment
        4. Residual autocorrelation

        These plots help assess:
        - Whether residuals are random (white noise)
        - Whether residuals are normally distributed
        - Whether there are remaining patterns in residuals
        - Model adequacy

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> viz.plot_residual_analysis('VJRQ+2M|RMJJ+F4')
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        from scipy import stats

        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Sort by timestamp and set as index
        route_data = route_data.sort_values('timestamp')
        route_data = route_data.set_index('timestamp')

        # Check for sufficient data
        min_required_hours = 2 * 24 * 7  # 2 weeks
        if len(route_data) < min_required_hours:
            print(f"Insufficient data for decomposition. Need at least {min_required_hours} hours.")
            return

        # Resample to hourly frequency
        route_data_hourly = route_data['avg_speed'].resample('h').mean()
        route_data_hourly = route_data_hourly.ffill(limit=3).bfill(limit=3).dropna()

        if len(route_data_hourly) < min_required_hours:
            print(f"Insufficient data after resampling.")
            return

        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                route_data_hourly,
                model='additive',
                period=24 * 7,
                extrapolate_trend='freq'
            )

            residuals = decomposition.resid.dropna()

            # Get route color and label
            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code)

            # Create figure with 4 subplots
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # Subplot 1: Residual time series
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(residuals.index, residuals.values, color=route_color,
                    linewidth=1, alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Residual (km/h)', fontsize=11, fontweight='bold')
            ax1.set_title('Residual Time Series', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Format x-axis
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Subplot 2: Histogram with normal distribution overlay
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(residuals, bins=30, density=True, alpha=0.7,
                    color=route_color, edgecolor='black')

            # Overlay normal distribution
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                    label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

            ax2.set_xlabel('Residual (km/h)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            # Subplot 3: Q-Q plot
            ax3 = fig.add_subplot(gs[1, 1])
            stats.probplot(residuals, dist="norm", plot=ax3)
            ax3.get_lines()[0].set_color(route_color)
            ax3.get_lines()[0].set_markersize(4)
            ax3.get_lines()[1].set_color('red')
            ax3.get_lines()[1].set_linewidth(2)
            ax3.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add overall title
            fig.suptitle(f'Residual Analysis: {route_label}',
                        fontsize=14, fontweight='bold', y=0.995)

            # Add statistics box
            # Perform normality test
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)

            stats_text = (
                f'Residual Statistics:\n'
                f'Mean: {mu:.4f} km/h\n'
                f'Std Dev: {sigma:.4f} km/h\n'
                f'Shapiro-Wilk p-value: {shapiro_p:.4f}\n'
                f'{"Normally distributed" if shapiro_p > 0.05 else "Not normally distributed"}'
            )
            fig.text(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            plt.tight_layout(rect=[0, 0.05, 1, 0.99])
            plt.show()

        except Exception as e:
            print(f"Error performing residual analysis: {e}")
            return

    # ========================================================================
    # Predictive Insight Visualizations
    # ========================================================================

    def plot_forecast(self, route_code: str, forecast_hours: int = 24) -> None:
        """
        Generate forecast visualization showing predicted speeds for next N hours.

        Creates a forecast plot using historical hour-of-day and day-of-week patterns
        to predict future speeds. Includes confidence interval bands based on
        historical variability.

        The forecast uses a simple but effective approach:
        - For each future hour, compute the average speed for that hour/day-of-week
        - Confidence intervals based on historical standard deviation
        - Assumes patterns repeat (no trend modeling)

        Parameters
        ----------
        route_code : str
            Route identifier
        forecast_hours : int, default=24
            Number of hours to forecast into the future

        Examples
        --------
        >>> viz.plot_forecast('VJRQ+2M|RMJJ+F4', forecast_hours=24)
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Get the last timestamp in the data
        last_timestamp = route_data['timestamp'].max()

        # Compute historical averages by hour and day-of-week
        historical_stats = route_data.groupby(['hour', 'day_of_week'])['avg_speed'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        # Generate forecast timestamps
        forecast_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq='h'
        )

        # Create forecast DataFrame
        forecast_data = []
        for ts in forecast_timestamps:
            hour = ts.hour
            day_of_week = ts.day_name()

            # Look up historical average for this hour/day
            hist = historical_stats[
                (historical_stats['hour'] == hour) &
                (historical_stats['day_of_week'] == day_of_week)
            ]

            if not hist.empty:
                mean_speed = hist['mean'].iloc[0]
                std_speed = hist['std'].iloc[0]
                count = hist['count'].iloc[0]

                # Compute confidence interval (95% = 1.96 * std)
                # Adjust for sample size using standard error
                se = std_speed / np.sqrt(count) if count > 0 else std_speed
                ci_lower = mean_speed - 1.96 * se
                ci_upper = mean_speed + 1.96 * se
            else:
                # No historical data for this hour/day combination
                mean_speed = route_data['avg_speed'].mean()
                std_speed = route_data['avg_speed'].std()
                ci_lower = mean_speed - 1.96 * std_speed
                ci_upper = mean_speed + 1.96 * std_speed

            forecast_data.append({
                'timestamp': ts,
                'forecast': mean_speed,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        forecast_df = pd.DataFrame(forecast_data)

        # Get recent historical data for context (last 48 hours)
        recent_cutoff = last_timestamp - pd.Timedelta(hours=48)
        recent_data = route_data[route_data['timestamp'] >= recent_cutoff].copy()

        # Get route color and label
        route_color = self._get_route_color(route_code)
        route_label = self._get_route_label(route_code)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot recent historical data
        ax.plot(recent_data['timestamp'], recent_data['avg_speed'],
               color=route_color, linewidth=2, alpha=0.8, label='Historical', marker='o', markersize=4)

        # Plot forecast
        ax.plot(forecast_df['timestamp'], forecast_df['forecast'],
               color='red', linewidth=2, linestyle='--', alpha=0.8, label='Forecast', marker='s', markersize=4)

        # Plot confidence interval
        ax.fill_between(forecast_df['timestamp'],
                       forecast_df['ci_lower'],
                       forecast_df['ci_upper'],
                       color='red', alpha=0.2, label='95% Confidence Interval')

        # Add vertical line at forecast start
        ax.axvline(x=last_timestamp, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(last_timestamp, ax.get_ylim()[1], '  Forecast Start',
               rotation=90, va='top', ha='right', fontsize=10, color='gray')

        # Format plot
        ax.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Speed Forecast: {route_label}\n'
                    f'Next {forecast_hours} Hours (Based on Historical Patterns)',
                    fontsize=14, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        # Add note
        fig.text(0.5, 0.01,
                'Note: Forecast based on historical hour-of-day and day-of-week patterns. '
                'Confidence intervals reflect historical variability.',
                ha='center', fontsize=10, style='italic', color='gray', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    def plot_typical_day_profile(self, day_of_week: Optional[str] = None,
                                  route_codes: list[str] | None = None) -> None:
        """
        Generate typical day profile for each day-of-week with variance bands.

        Creates line plots showing the typical speed profile for each day of the week,
        with shaded regions showing variability (±1 standard deviation).

        If day_of_week is specified, shows only that day. Otherwise, shows all days
        as a single horizontal plot from Monday 12 AM to Sunday midnight.

        Parameters
        ----------
        day_of_week : str, optional
            Specific day to plot ('Monday', 'Tuesday', etc.). If None, plots all days.
        route_codes : list[str], optional
            Routes to include. If None, includes all routes.

        Examples
        --------
        >>> viz.plot_typical_day_profile('Monday')
        >>> viz.plot_typical_day_profile()  # All days
        >>> viz.plot_typical_day_profile(route_codes=['2HM2+P8|XJV5+RG'])
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        if day_of_week:
            # --- Single-day plot (unch behaviour) ---
            fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
            day_data = df[df['day_of_week'] == day_of_week]

            if day_data.empty:
                ax.text(0.5, 0.5, f'No data for {day_of_week}',
                        ha='center', va='center', transform=ax.transAxes)
            else:
                for route_code in self.routes:
                    route_day_data = day_data[day_data['route_code'] == route_code]
                    if route_day_data.empty:
                        continue
                    hourly_stats = route_day_data.groupby('hour')['avg_speed'].agg([
                        ('mean', 'mean'), ('std', 'std')
                    ]).reset_index()
                    hourly_stats = hourly_stats.set_index('hour').reindex(range(24)).reset_index()
                    route_color = self._get_route_color(route_code)
                    route_label = self._get_route_label(route_code, 'short')
                    ax.plot(hourly_stats['hour'], hourly_stats['mean'],
                            color=route_color, linewidth=2, label=route_label, alpha=0.9)
                    valid_mask = hourly_stats['mean'].notna()
                    if valid_mask.any():
                        hours = hourly_stats.loc[valid_mask, 'hour']
                        means = hourly_stats.loc[valid_mask, 'mean']
                        stds = hourly_stats.loc[valid_mask, 'std'].fillna(0)
                        ax.fill_between(hours, means - stds, means + stds,
                                        color=route_color, alpha=0.15, linewidth=0)

            ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
            ax.set_ylabel('Avg Speed (km/h)', fontsize=11, fontweight='bold')
            ax.set_title(f'Typical Day Profile: {day_of_week}\n(Shaded regions show ±1 standard deviation)',
                         fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels([self._format_hour_label(h) for h in range(0, 24, 3)],
                               rotation=45, ha='right')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()
            return

        # --- All-days: single horizontal plot, Monday 12 AM → Sunday midnight ---
        selected_routes = self.routes if route_codes is None else [r for r in self.routes if r in set(route_codes)]

        if not selected_routes:
            warnings.warn("No valid routes selected for typical day profile plot.")
            return

        fig, ax = plt.subplots(figsize=(20, 14), dpi=150)

        day_offset = {day: i * 24 for i, day in enumerate(days_of_week)}

        for route_code in selected_routes:
            route_data = df[df['route_code'] == route_code]
            if route_data.empty:
                continue

            route_color = self._get_route_color(route_code)
            route_label = self._get_route_label(route_code, 'short')

            all_hours = []
            all_means = []
            all_stds = []

            for day in days_of_week:
                day_data = route_data[route_data['day_of_week'] == day]
                if day_data.empty:
                    all_hours.extend([day_offset[day] + h for h in range(24)])
                    all_means.extend([np.nan] * 24)
                    all_stds.extend([np.nan] * 24)
                    continue
                hourly_stats = day_data.groupby('hour')['avg_speed'].agg([
                    ('mean', 'mean'), ('std', 'std')
                ]).reset_index()
                hourly_stats = hourly_stats.set_index('hour').reindex(range(24)).reset_index()
                for h in range(24):
                    all_hours.append(day_offset[day] + h)
                    all_means.append(hourly_stats.loc[h, 'mean'])
                    all_stds.append(hourly_stats.loc[h, 'std'])

            all_hours = np.array(all_hours)
            all_means = np.array(all_means, dtype=float)
            all_stds = np.array(all_stds, dtype=float)

            valid_mask = ~np.isnan(all_means)
            if valid_mask.sum() >= 4:
                vh = all_hours[valid_mask]
                vm = all_means[valid_mask]
                vs = np.nan_to_num(all_stds[valid_mask])
                f_mean = interp1d(vh, vm, kind='cubic', bounds_error=False, fill_value='extrapolate')
                f_std = interp1d(vh, vs, kind='linear', bounds_error=False, fill_value='extrapolate')
                smooth_hours = np.linspace(vh.min(), vh.max(), len(vh) * 5)
                smooth_means = f_mean(smooth_hours)
                smooth_stds = f_std(smooth_hours)
                ax.plot(smooth_hours, smooth_means,
                        color=route_color, linewidth=2, label=route_label, alpha=0.9)
                ax.fill_between(smooth_hours, smooth_means - smooth_stds, smooth_means + smooth_stds,
                                color=route_color, alpha=0.12, linewidth=0)
            else:
                ax.plot(all_hours, all_means,
                        color=route_color, linewidth=2, label=route_label, alpha=0.9)
                if valid_mask.any():
                    vh = all_hours[valid_mask]
                    vm = all_means[valid_mask]
                    vs = np.nan_to_num(all_stds[valid_mask])
                    ax.fill_between(vh, vm - vs, vm + vs,
                                    color=route_color, alpha=0.12, linewidth=0)

        # X-axis: day boundaries with labels, hour ticks every 3 hours
        ax.set_xlim(0, 168)
        ax.set_xticks(range(0, 169, 3))
        hour_labels = []
        for tick in range(0, 169, 3):
            day_idx = tick // 24
            hour = tick % 24
            if hour == 0 and day_idx < 7:
                hour_labels.append(days_of_week[day_idx][:3])
            else:
                hour_labels.append(f'{hour:02d}')
        ax.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=9)

        # Vertical separators between days
        for i in range(1, 7):
            ax.axvline(x=i * 24, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Day name labels at the top of each day section
        for i, day in enumerate(days_of_week):
            ax.text(i * 24 + 12, 1.02, day, ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.7))

        ax.set_xlabel('Day & Hour of Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title('Typical Week Profile (Mon 12 AM → Sun Midnight)',
                     fontsize=14, fontweight='bold', pad=80)
        fig.text(0.5, 0.89, '(Shaded regions show ±1 standard deviation)',
                 fontsize=11, fontweight='normal', ha='center', va='center')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.15)
        plt.show()

    def plot_current_vs_predicted(self, route_code: str, reference_date: Optional[str] = None) -> None:
        """
        Identify and visualize deviations from expected performance.

        Compares actual speeds on a specific date against the typical pattern
        for that day-of-week, highlighting periods where performance deviates
        significantly from expectations.

        Parameters
        ----------
        route_code : str
            Route identifier
        reference_date : str, optional
            Date to analyze (format: 'YYYY-MM-DD'). If None, uses most recent date.

        Examples
        --------
        >>> viz.plot_current_vs_predicted('VJRQ+2M|RMJJ+F4', '2025-09-15')
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Determine reference date
        if reference_date:
            ref_date = pd.to_datetime(reference_date).date()
        else:
            ref_date = route_data['timestamp'].max().date()

        # Get data for reference date
        current_data = route_data[route_data['timestamp'].dt.date == ref_date].copy()

        if current_data.empty:
            print(f"No data available for date: {ref_date}")
            return

        # Get day of week for reference date
        day_of_week = pd.to_datetime(ref_date).day_name()

        # Compute typical pattern for this day-of-week (excluding reference date)
        typical_data = route_data[
            (route_data['day_of_week'] == day_of_week) &
            (route_data['timestamp'].dt.date != ref_date)
        ]

        typical_pattern = typical_data.groupby('hour')['avg_speed'].agg([
            ('mean', 'mean'),
            ('std', 'std')
        ]).reset_index()

        # Merge current with typical
        current_data = current_data.merge(
            typical_pattern,
            on='hour',
            how='left',
            suffixes=('', '_typical')
        )

        current_data['deviation'] = current_data['avg_speed'] - current_data['mean']
        current_data['is_anomaly'] = current_data['deviation'].abs() > 2 * current_data['std']

        # Get route color and label
        route_color = self._get_route_color(route_code)
        route_label = self._get_route_label(route_code)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Subplot 1: Current vs Typical
        ax1.plot(current_data['hour'], current_data['avg_speed'],
                color=route_color, linewidth=2.5, marker='o', markersize=6,
                label=f'Actual ({ref_date})', alpha=0.9)
        ax1.plot(typical_pattern['hour'], typical_pattern['mean'],
                color='gray', linewidth=2, linestyle='--', marker='s', markersize=5,
                label=f'Typical {day_of_week}', alpha=0.7)

        # Add typical variance band
        ax1.fill_between(typical_pattern['hour'],
                        typical_pattern['mean'] - typical_pattern['std'],
                        typical_pattern['mean'] + typical_pattern['std'],
                        color='gray', alpha=0.2, label='±1σ Range')

        # Highlight anomalies
        anomalies = current_data[current_data['is_anomaly']]
        if not anomalies.empty:
            ax1.scatter(anomalies['hour'], anomalies['avg_speed'],
                       color='red', s=150, marker='X', edgecolors='darkred',
                       linewidths=2, zorder=5, label=f'Anomalies (n={len(anomalies)})')

        ax1.set_ylabel('Speed (km/h)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Current vs Predicted: {route_label}',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='best', framealpha=0.95, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Subplot 2: Deviation
        colors = ['red' if x > 0 else 'blue' for x in current_data['deviation']]
        ax2.bar(current_data['hour'], current_data['deviation'],
               color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

        # Add threshold lines
        if not typical_pattern['std'].isna().all():
            avg_std = typical_pattern['std'].mean()
            ax2.axhline(y=2*avg_std, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='±2σ Threshold')
            ax2.axhline(y=-2*avg_std, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Deviation (km/h)', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xticklabels([self._format_hour_label(h) for h in range(0, 24, 2)],
                           rotation=45, ha='right')
        ax2.legend(loc='best', framealpha=0.95, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add interpretation note
        fig.text(0.5, 0.01,
                f'Interpretation: Red bars = faster than typical. Blue bars = slower than typical. '
                f'X markers = significant deviations (>2σ).',
                ha='center', fontsize=10, style='italic', color='gray', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()

    def plot_seasonal_trends(self, route_code: str) -> None:
        """
        Generate month-over-month comparison showing seasonal trends.

        Creates visualizations showing how route performance changes across months,
        revealing seasonal patterns and long-term trends.

        Parameters
        ----------
        route_code : str
            Route identifier

        Examples
        --------
        >>> viz.plot_seasonal_trends('VJRQ+2M|RMJJ+F4')
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Filter data for this route
        route_data = df[df['route_code'] == route_code].copy()

        if route_data.empty:
            print(f"No data available for route: {self._get_route_label(route_code)}")
            return

        # Add month name
        route_data['month_name'] = route_data['timestamp'].dt.strftime('%Y-%m')

        # Compute monthly statistics
        monthly_stats = route_data.groupby('month_name')['avg_speed'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        # Sort by month
        monthly_stats = monthly_stats.sort_values('month_name')

        # Get route color and label
        route_color = self._get_route_color(route_code)
        route_label = self._get_route_label(route_code)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot monthly averages
        ax.plot(range(len(monthly_stats)), monthly_stats['mean'],
               color=route_color, linewidth=2.5, marker='o', markersize=8,
               label='Monthly Average', alpha=0.9)

        # Add error bars
        ax.errorbar(range(len(monthly_stats)), monthly_stats['mean'],
                   yerr=monthly_stats['std'], fmt='none',
                   ecolor=route_color, alpha=0.3, capsize=5, capthick=2)

        # Format plot
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Seasonal Trends: {route_label}\n'
                    f'Month-over-Month Performance',
                    fontsize=14, fontweight='bold', pad=20)

        # Set x-axis labels
        ax.set_xticks(range(len(monthly_stats)))
        ax.set_xticklabels(monthly_stats['month_name'], rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # Add legend
        ax.legend(loc='best', framealpha=0.95, fontsize=10)

        # Add sample size annotations
        for i, row in monthly_stats.iterrows():
            ax.text(i, row['mean'], f"  n={int(row['count'])}",
                   fontsize=8, va='bottom', ha='left', color='gray', style='italic')

        plt.tight_layout()
        plt.show()

    def plot_lag_correlations(self) -> None:
        """
        Show how one route's performance predicts another's.

        Computes cross-correlations between routes at different time lags,
        revealing which routes tend to experience similar conditions with
        a time delay.

        Examples
        --------
        >>> viz.plot_lag_correlations()
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Create pivot table: rows = timestamps, columns = routes
        pivot_data = df.pivot_table(
            index='timestamp',
            columns='route_code',
            values='avg_speed',
            aggfunc='mean'
        )

        # Compute cross-correlations at different lags
        max_lag = 24  # Maximum lag in hours
        n_routes = len(self.routes)

        # Create figure with subplots for each route pair
        fig, axes = plt.subplots(n_routes, n_routes, figsize=(16, 16))

        for i, route1 in enumerate(self.routes):
            for j, route2 in enumerate(self.routes):
                ax = axes[i, j]

                if route1 not in pivot_data.columns or route2 not in pivot_data.columns:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: show route label
                    ax.text(0.5, 0.5, self._get_route_label(route1, 'short'),
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)
                else:
                    # Compute cross-correlation
                    series1 = pivot_data[route1].dropna()
                    series2 = pivot_data[route2].dropna()

                    # Align series
                    common_idx = series1.index.intersection(series2.index)
                    series1 = series1.loc[common_idx]
                    series2 = series2.loc[common_idx]

                    if len(series1) < max_lag:
                        ax.set_visible(False)
                        continue

                    # Compute correlations at different lags
                    lags = range(-max_lag, max_lag + 1)
                    correlations = []

                    for lag in lags:
                        if lag == 0:
                            corr = series1.corr(series2)
                        elif lag > 0:
                            corr = series1[:-lag].corr(series2[lag:])
                        else:
                            corr = series1[-lag:].corr(series2[:lag])
                        correlations.append(corr)

                    # Plot
                    color1 = self._get_route_color(route1)
                    ax.plot(lags, correlations, color=color1, linewidth=1.5, alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

                    # Highlight maximum correlation
                    max_corr_idx = np.argmax(np.abs(correlations))
                    max_lag = lags[max_corr_idx]
                    max_corr = correlations[max_corr_idx]
                    ax.plot(max_lag, max_corr, 'ro', markersize=4)

                    # Format
                    ax.set_ylim(-1, 1)
                    if i == n_routes - 1:
                        ax.set_xlabel('Lag (hours)', fontsize=8)
                    if j == 0:
                        ax.set_ylabel('Correlation', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.2)

        fig.suptitle('Cross-Correlation Matrix: Route Lag Analysis\n'
                    '(Shows how one route predicts another at different time lags)',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    def plot_best_travel_times(self) -> None:
        """
        Show optimal departure times for each route.

        Identifies and visualizes the best times to travel on each route
        based on historical speed data. Shows:
        - Best hour of day for each route
        - Worst hour of day for each route
        - Speed distribution throughout the day

        Examples
        --------
        >>> viz.plot_best_travel_times()
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)

        # Compute average speed by route and hour
        hourly_speeds = df.groupby(['route_code', 'hour'])['avg_speed'].mean().reset_index()

        # Find best and worst hours for each route
        best_times = []
        for route_code in self.routes:
            route_hourly = hourly_speeds[hourly_speeds['route_code'] == route_code]

            if route_hourly.empty:
                continue

            best_hour = route_hourly.loc[route_hourly['avg_speed'].idxmax(), 'hour']
            best_speed = route_hourly['avg_speed'].max()
            worst_hour = route_hourly.loc[route_hourly['avg_speed'].idxmin(), 'hour']
            worst_speed = route_hourly['avg_speed'].min()

            best_times.append({
                'route_code': route_code,
                'route_label': self._get_route_label(route_code, 'short'),
                'best_hour': int(best_hour),
                'best_speed': best_speed,
                'worst_hour': int(worst_hour),
                'worst_speed': worst_speed,
                'speed_range': best_speed - worst_speed
            })

        best_times_df = pd.DataFrame(best_times)

        # Sort by speed range (most variable first)
        best_times_df = best_times_df.sort_values('speed_range', ascending=False)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Subplot 1: Best vs Worst hours
        x_pos = np.arange(len(best_times_df))
        width = 0.35

        # Plot best hours
        colors_best = [self._get_route_color(r) for r in best_times_df['route_code']]
        ax1.barh(x_pos - width/2, best_times_df['best_speed'],
                width, label='Best Hour', color=colors_best, alpha=0.8, edgecolor='black')

        # Plot worst hours
        ax1.barh(x_pos + width/2, best_times_df['worst_speed'],
                width, label='Worst Hour', color=colors_best, alpha=0.3, edgecolor='black')

        # Add hour labels
        for i, row in best_times_df.iterrows():
            idx = best_times_df.index.get_loc(i)
            ax1.text(row['best_speed'], idx - width/2,
                    f"  {self._format_hour_label(row['best_hour'])}",
                    va='center', fontsize=8, fontweight='bold')
            ax1.text(row['worst_speed'], idx + width/2,
                    f"  {self._format_hour_label(row['worst_hour'])}",
                    va='center', fontsize=8, style='italic', color='gray')

        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(best_times_df['route_label'])
        ax1.set_xlabel('Average Speed (km/h)', fontsize=11, fontweight='bold')
        ax1.set_title('Best vs Worst Travel Times by Route',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.95)
        ax1.grid(True, alpha=0.3, axis='x')

        # Subplot 2: Hourly heatmap showing all routes
        pivot_hourly = hourly_speeds.pivot(index='route_code', columns='hour', values='avg_speed')

        # Reorder rows to match best_times_df
        pivot_hourly = pivot_hourly.reindex(best_times_df['route_code'])

        # Create heatmap
        im = ax2.imshow(pivot_hourly.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')

        # Set ticks
        ax2.set_yticks(range(len(best_times_df)))
        ax2.set_yticklabels(best_times_df['route_label'], fontsize=9)
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xticklabels([self._format_hour_label(h) for h in range(0, 24, 2)],
                           rotation=45, ha='right', fontsize=9)

        ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax2.set_title('Speed Heatmap: All Routes by Hour\n(Green = Fast, Red = Slow)',
                     fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Avg Speed (km/h)', fontsize=10)

        # Add overall title
        fig.suptitle('Optimal Travel Times Analysis',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Print summary table
        print("\nBest Travel Times Summary:")
        print("=" * 80)
        print(f"{'Route':<25} {'Best Time':<15} {'Speed':<12} {'Worst Time':<15} {'Speed':<12}")
        print("=" * 80)
        for _, row in best_times_df.iterrows():
            print(f"{row['route_label']:<25} "
                  f"{self._format_hour_label(row['best_hour']):<15} "
                  f"{row['best_speed']:>6.1f} km/h   "
                  f"{self._format_hour_label(row['worst_hour']):<15} "
                  f"{row['worst_speed']:>6.1f} km/h")
        print("=" * 80)

    def create_route_checkbox_grid(self) -> dict:
        """
        Create an interactive route selection checkbox grid with select/deselect all toggle.

        Returns a dict containing:
        - ``output``: ipywidgets.Output for rendering plots
        - ``controls``: HBox containing the toggle button
        - ``grid``: GridBox of route checkboxes
        - ``get_selected``: callable returning list of selected route codes
        - ``render``: callable taking a plot_fn(selected_routes) -> None that renders into the output

        The ``render`` method wires up all checkbox/toggle event handlers automatically.
        The caller only needs to provide ``plot_fn`` and display the widgets.

        Examples
        --------
        >>> selector = viz.create_route_checkbox_grid()
        >>> def plot_fn(routes):
        ...     if routes:
        ...         viz.plot_hour_of_day_profiles(routes)
        >>> selector['render'](plot_fn)
        >>> display(selector['output'], selector['controls'], selector['grid'])
        """
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        route_checkboxes = []
        route_items = []
        for route_code in self.routes:
            color = self._get_route_color(route_code)
            cb = widgets.Checkbox(
                value=True,
                description=self._get_route_label(route_code),
                indent=False,
                layout=widgets.Layout(width='auto')
            )
            route_checkboxes.append(cb)
            swatch = widgets.HTML(
                value=f'<div style="width:14px;height:14px;background-color:{color};'
                      f'border:1px solid #555;border-radius:2px;display:inline-block;"></div>',
                layout=widgets.Layout(margin='2px 4px 0 0')
            )
            route_items.append(
                widgets.HBox([swatch, cb], layout=widgets.Layout(width='auto'))
            )

        select_toggle = widgets.Button(
            description='Deselect all',
            tooltip='Select all or deselect all routes',
            button_style=''
        )

        checkbox_grid = widgets.GridBox(
            children=route_items,
            layout=widgets.Layout(
                grid_template_columns='repeat(5, minmax(180px, 1fr))',
                grid_gap='8px 16px',
                width='100%'
            )
        )

        controls = widgets.HBox([select_toggle])
        out = widgets.Output()
        _updating = [False]

        def get_selected():
            return [rc for rc, cb in zip(self.routes, route_checkboxes) if cb.value]

        def refresh_toggle_label():
            count = sum(cb.value for cb in route_checkboxes)
            if count == len(route_checkboxes):
                select_toggle.description = 'Deselect all'
            elif count == 0:
                select_toggle.description = 'Select all'
            else:
                select_toggle.description = 'Select all'

        def make_render(plot_fn):
            def render():
                selected = get_selected()
                with out:
                    clear_output(wait=True)
                    plot_fn(selected)
                refresh_toggle_label()
            return render

        def on_checkbox_change(change, render_fn):
            if _updating[0]:
                return
            if change['type'] == 'change' and change['name'] == 'value':
                render_fn()

        def on_toggle_click(_, render_fn):
            if _updating[0]:
                return
            target = sum(cb.value for cb in route_checkboxes) != len(route_checkboxes)
            _updating[0] = True
            try:
                for cb in route_checkboxes:
                    cb.unobserve_all()
                    cb.value = target
                render_fn()
            finally:
                _updating[0] = False
            # Re-attach observers after bulk update
            for cb in route_checkboxes:
                cb.observe(lambda ch, rf=render_fn: on_checkbox_change(ch, rf), names='value')

        def render(plot_fn):
            render_fn = make_render(plot_fn)
            for cb in route_checkboxes:
                cb.observe(lambda ch, rf=render_fn: on_checkbox_change(ch, rf), names='value')
            select_toggle.on_click(lambda _, rf=render_fn: on_toggle_click(_, rf))
            render_fn()

        return {
            'output': out,
            'controls': controls,
            'grid': checkbox_grid,
            'get_selected': get_selected,
            'render': render,
        }


    def create_linked_plots(self, route_codes: Optional[List[str]] = None) -> 'plotly.graph_objs.Figure':
        """
        Create linked interactive plots where selecting time range highlights data across plots.

        Uses Plotly to create interactive visualizations with:
        - Hover tooltips showing detailed information
        - Linked brushing across multiple subplots
        - Zoom and pan synchronization

        Parameters
        ----------
        route_codes : list of str, optional
            List of route codes to include. If None, includes all routes.

        Returns
        -------
        plotly.graph_objs.Figure
            Interactive figure with linked plots

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> fig = viz.create_linked_plots(['ROUTE_A', 'ROUTE_B'])
        >>> fig.show()
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)
        
        # Filter routes if specified
        if route_codes is None:
            route_codes = self.routes[:min(5, len(self.routes))]  # Limit to 5 routes for readability
        
        # Filter data
        plot_data = df[df['route_code'].isin(route_codes)].copy()
        
        if plot_data.empty:
            print("No data available for specified routes")
            return None
        
        # Create subplots: time series, distribution, hour-of-day pattern
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Speed Over Time (Linked Brushing)',
                'Speed Distribution',
                'Average Speed by Hour of Day'
            ),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Plot 1: Time series with hover tooltips
        for route_code in route_codes:
            route_data = plot_data[plot_data['route_code'] == route_code].sort_values('timestamp')
            route_label = self._get_route_label(route_code)
            route_color = self.color_palette.get(route_code, '#999999')
            
            # Create hover text with detailed information
            hover_text = []
            for _, row in route_data.iterrows():
                text = (
                    f"<b>{route_label}</b><br>"
                    f"Time: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Speed: {row['avg_speed']:.1f} km/h<br>"
                    f"Duration: {row['duration']:.1f} min<br>"
                    f"Distance: {row['distance']:.1f} km"
                )
                if 'percentile_rank' in row:
                    text += f"<br>Percentile: {row['percentile_rank']:.1f}%"
                hover_text.append(text)
            
            fig.add_trace(
                go.Scatter(
                    x=route_data['timestamp'],
                    y=route_data['avg_speed'],
                    mode='lines+markers',
                    name=route_label,
                    line=dict(color=route_color, width=2),
                    marker=dict(size=4, color=route_color),
                    hovertext=hover_text,
                    hoverinfo='text',
                    legendgroup=route_code
                ),
                row=1, col=1
            )
        
        # Plot 2: Distribution (box plot)
        for route_code in route_codes:
            route_data = plot_data[plot_data['route_code'] == route_code]
            route_label = self._get_route_label(route_code)
            route_color = self.color_palette.get(route_code, '#999999')
            
            fig.add_trace(
                go.Box(
                    y=route_data['avg_speed'],
                    name=route_label,
                    marker_color=route_color,
                    boxmean='sd',
                    hoverinfo='y',
                    legendgroup=route_code,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 3: Hour-of-day pattern
        for route_code in route_codes:
            route_data = plot_data[plot_data['route_code'] == route_code]
            route_label = self._get_route_label(route_code)
            route_color = self.color_palette.get(route_code, '#999999')
            
            # Compute hourly averages
            hourly_avg = route_data.groupby('hour')['avg_speed'].agg(['mean', 'std']).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg['hour'],
                    y=hourly_avg['mean'],
                    mode='lines+markers',
                    name=route_label,
                    line=dict(color=route_color, width=2),
                    marker=dict(size=6, color=route_color),
                    error_y=dict(
                        type='data',
                        array=hourly_avg['std'],
                        visible=True,
                        color=route_color,
                        thickness=1.5,
                        width=3
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Hour: %{x}<br>Speed: %{y:.1f} km/h<extra></extra>',
                    legendgroup=route_code,
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Route", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1, dtick=2)
        
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=2, col=1)
        fig.update_yaxes(title_text="Avg Speed (km/h)", row=3, col=1)
        
        fig.update_layout(
            height=1000,
            title_text="Interactive Linked Plots - Traffic Analysis",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

    def add_hover_tooltips(self, fig: 'matplotlib.figure.Figure', 
                          data: pd.DataFrame) -> 'matplotlib.figure.Figure':
        """
        Add hover tooltips to matplotlib figure (note: limited interactivity).

        Note: Matplotlib has limited interactive capabilities compared to Plotly.
        For full interactive tooltips, use create_linked_plots() with Plotly instead.

        This method adds basic annotation support using mplcursors if available.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to enhance
        data : pd.DataFrame
            Data associated with the plot

        Returns
        -------
        matplotlib.figure.Figure
            Enhanced figure with tooltip support

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data['x'], data['y'])
        >>> fig = viz.add_hover_tooltips(fig, data)
        """
        try:
            import mplcursors
            
            # Add cursor support to all axes in the figure
            for ax in fig.get_axes():
                cursor = mplcursors.cursor(ax, hover=True)
                
                @cursor.connect("add")
                def on_add(sel):
                    # Customize tooltip appearance
                    sel.annotation.set_bbox(dict(boxstyle='round,pad=0.5', 
                                                 facecolor='yellow', 
                                                 alpha=0.9,
                                                 edgecolor='black'))
                    sel.annotation.set_fontsize(9)
            
            return fig
        except ImportError:
            warnings.warn(
                "mplcursors not installed. Install with 'pip install mplcursors' "
                "for tooltip support. For better interactivity, use create_linked_plots() instead."
            )
            return fig


    def create_summary_table(self, route_codes: Optional[List[str]] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            aggregation: str = 'D') -> pd.DataFrame:
        """
        Create dynamic summary statistics table with filter selections.

        Generates a comprehensive summary table that updates based on:
        - Selected routes
        - Date range filter
        - Aggregation level (hourly, daily, weekly)

        Parameters
        ----------
        route_codes : list of str, optional
            List of route codes to include. If None, includes all routes.
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        aggregation : str, default='D'
            Aggregation level: 'H' (hourly), 'D' (daily), 'W' (weekly)

        Returns
        -------
        pd.DataFrame
            Summary statistics table with columns:
            - route_code, route_label
            - observations, avg_speed, std_speed
            - min_speed, max_speed, median_speed
            - p25_speed, p75_speed (quartiles)
            - avg_duration, completeness_pct

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> summary = viz.create_summary_table(
        ...     route_codes=['ROUTE_A', 'ROUTE_B'],
        ...     start_date='2024-01-01',
        ...     end_date='2024-01-31',
        ...     aggregation='D'
        ... )
        >>> print(summary)
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)
        
        # Filter by routes
        if route_codes is None:
            route_codes = self.routes
        filtered_df = df[df['route_code'].isin(route_codes)].copy()
        
        # Filter by date range
        if start_date:
            filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(end_date)]
        
        if filtered_df.empty:
            print("No data available for specified filters")
            return pd.DataFrame()
        
        # Aggregate data if needed
        if aggregation != 'H':
            # Resample to specified aggregation level
            agg_data = []
            for route_code in route_codes:
                route_data = filtered_df[filtered_df['route_code'] == route_code].copy()
                if not route_data.empty:
                    route_data = route_data.set_index('timestamp')
                    resampled = route_data.resample(aggregation).agg({
                        'avg_speed': 'mean',
                        'duration': 'mean',
                        'distance': 'mean'
                    }).reset_index()
                    resampled['route_code'] = route_code
                    agg_data.append(resampled)
            
            if agg_data:
                filtered_df = pd.concat(agg_data, ignore_index=True)
        
        # Compute summary statistics per route
        summary_list = []
        for route_code in route_codes:
            route_data = filtered_df[filtered_df['route_code'] == route_code]
            
            if route_data.empty:
                continue
            
            # Get route label
            route_label = self._get_route_label(route_code)
            
            # Compute statistics
            summary = {
                'route_code': route_code,
                'route_label': route_label,
                'observations': len(route_data),
                'avg_speed': route_data['avg_speed'].mean(),
                'std_speed': route_data['avg_speed'].std(),
                'min_speed': route_data['avg_speed'].min(),
                'max_speed': route_data['avg_speed'].max(),
                'median_speed': route_data['avg_speed'].median(),
                'p25_speed': route_data['avg_speed'].quantile(0.25),
                'p75_speed': route_data['avg_speed'].quantile(0.75),
                'avg_duration': route_data['duration'].mean(),
                'completeness_pct': (len(route_data) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            }
            
            summary_list.append(summary)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_list)
        
        # Round numeric columns
        numeric_cols = ['avg_speed', 'std_speed', 'min_speed', 'max_speed', 
                       'median_speed', 'p25_speed', 'p75_speed', 'avg_duration', 'completeness_pct']
        for col in numeric_cols:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(2)
        
        return summary_df

    def export_report_template(self, output_path: str,
                              route_codes: Optional[List[str]] = None,
                              include_visualizations: bool = True) -> str:
        """
        Generate downloadable report template capturing visualization states.

        Creates a comprehensive HTML report with:
        - Summary statistics tables
        - Embedded visualizations (if requested)
        - Analysis metadata (date range, filters, etc.)
        - Exportable format (HTML, can be converted to PDF)

        Parameters
        ----------
        output_path : str
            Path to save the report (e.g., 'report.html')
        route_codes : list of str, optional
            List of route codes to include. If None, includes all routes.
        include_visualizations : bool, default=True
            Whether to include embedded visualizations in the report

        Returns
        -------
        str
            Path to the generated report file

        Examples
        --------
        >>> viz = VisualizationEngine(df, routes_df)
        >>> report_path = viz.export_report_template(
        ...     'traffic_report.html',
        ...     route_codes=['ROUTE_A', 'ROUTE_B'],
        ...     include_visualizations=True
        ... )
        >>> print(f"Report saved to: {report_path}")
        """
        # Ensure temporal features exist
        df = self._ensure_temporal_features(self.df)
        
        # Filter routes
        if route_codes is None:
            route_codes = self.routes
        
        # Create summary table
        summary_df = self.create_summary_table(route_codes=route_codes)
        
        # Get date range
        min_date = df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d')
        
        # Start building HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 15px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 10px;
        }}
        .visualization {{
            margin-top: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Traffic Analysis Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Report Metadata</h2>
        <p><strong>Date Range:</strong> {min_date} to {max_date}</p>
        <p><strong>Routes Analyzed:</strong> {len(route_codes)}</p>
        <p><strong>Total Observations:</strong> {len(df):,}</p>
    </div>
    
    <div class="section">
        <h2>Summary Statistics</h2>
        {summary_df.to_html(index=False, classes='summary-table')}
    </div>
"""
        
        # Add visualizations if requested
        if include_visualizations:
            html_content += """
    <div class="section">
        <h2>Visualizations</h2>
        <p class="metadata">Note: Interactive visualizations are best viewed in Jupyter notebooks. 
        This report contains static snapshots.</p>
        <div class="visualization">
            <p><em>Visualizations can be generated separately using the VisualizationEngine methods.</em></p>
        </div>
    </div>
"""
        
        # Add footer
        html_content += """
    <div class="section metadata">
        <p><strong>Analysis Tool:</strong> Traffic Analysis Enhancement System</p>
        <p><strong>Report Format:</strong> HTML (can be converted to PDF using browser print function)</p>
    </div>
</body>
</html>
"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report successfully exported to: {output_path}")
        return output_path

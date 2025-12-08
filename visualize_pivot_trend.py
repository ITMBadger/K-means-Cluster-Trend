# visualize_pivot_trend.py
"""
Pivot Cluster Trend Visualization - Interactive Plotly chart showing OHLC with 
colored background zones:
- Green background: pivot_bull (uptrend)  
- Red background: pivot_bear (downtrend)
- No color: neutral/undefined trend
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import jit, prange
import talib as ta
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import data utilities
from data_utils import fetch_intraday_data, resample_to_timeframe


# =============================================================================
# VISUALIZATION CONFIG - EDIT THESE PARAMETERS
# =============================================================================

# Data Parameters
SYMBOL = "QQQ"                    # Stock ticker symbol
YEARS_BACK = 6.0                  # Years of historical data
BASE_TIMEFRAME = "1min"           # Base timeframe for fetching/caching
DISPLAY_TIMEFRAME = "5min"        # Timeframe for visualization

# Display Options
SHOW_PIVOTS = True                # Show pivot high/low markers
SHOW_HURST = False                # Show Hurst exponent subplot
CHART_HEIGHT = 800                # Chart height in pixels

# Display date range (set to None to show all data)
# Format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' or None
DISPLAY_START_DATE = '2025-09-01'   # e.g., '2024-06-01' or '2024-06-01 09:30:00'
DISPLAY_END_DATE = '2025-12-01'     # e.g., '2024-12-01' or None for latest data

# Cache Options
USE_CACHE = True                  # Use cached data if available

# Export Options
SAVE_HTML = False                 # Save chart as HTML file
OUTPUT_DIR = None                 # Output directory (None = ./charts)


# =============================================================================
# PIVOT TREND CONFIGURATION
# =============================================================================

@dataclass
class PivotTrendConfig:
    """Configuration for pivot cluster trend calculation - matches strategy_core.py."""
    
    # ATR for adaptive fractal N
    ATR_PERIOD: int = 14
    ATR_NORM_WINDOW: int = 100
    
    # Hurst Exponent
    HURST_WINDOW: int = 100
    HURST_USE_MULTISCALE: bool = True
    
    # Fractals
    MIN_FRACTAL_N: int = 2
    MAX_FRACTAL_N: int = 5
    
    # Adaptive Lookback
    MIN_LOOKBACK: int = 50
    MAX_LOOKBACK: int = 200
    
    # Pivot Cluster Trend
    MIN_PIVOTS: int = 4
    TREND_PERSISTENCE: int = 5


# =============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def _calculate_rs_for_period(returns: np.ndarray, period_len: int) -> float:
    """Calculate R/S statistic for a given period length."""
    n = len(returns)
    if n < period_len or period_len < 8:
        return -1.0
    
    num_chunks = n // period_len
    if num_chunks == 0:
        return -1.0
    
    rs_sum = 0.0
    valid_chunks = 0
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * period_len
        end = start + period_len
        chunk = returns[start:end]
        
        mean_val = 0.0
        for i in range(period_len):
            mean_val += chunk[i]
        mean_val /= period_len
        
        cumsum = 0.0
        min_cumsum = 0.0
        max_cumsum = 0.0
        
        for i in range(period_len):
            cumsum += chunk[i] - mean_val
            if cumsum < min_cumsum:
                min_cumsum = cumsum
            if cumsum > max_cumsum:
                max_cumsum = cumsum
        
        R = max_cumsum - min_cumsum
        
        var_sum = 0.0
        for i in range(period_len):
            var_sum += (chunk[i] - mean_val) ** 2
        S = np.sqrt(var_sum / (period_len - 1))
        
        if S > 1e-10 and R > 1e-10:
            rs_sum += R / S
            valid_chunks += 1
    
    if valid_chunks == 0:
        return -1.0
    
    return rs_sum / valid_chunks


@jit(nopython=True, cache=True)
def _hurst_rs_single_fixed(prices: np.ndarray) -> float:
    """Calculate Hurst exponent via R/S analysis on LOG RETURNS."""
    n = len(prices)
    if n < 20:
        return 0.5
    
    returns = np.empty(n - 1)
    for i in range(n - 1):
        if prices[i] > 0 and prices[i + 1] > 0:
            returns[i] = np.log(prices[i + 1] / prices[i])
        else:
            returns[i] = 0.0
    
    n_ret = len(returns)
    if n_ret < 10:
        return 0.5
    
    mean_val = 0.0
    for i in range(n_ret):
        mean_val += returns[i]
    mean_val /= n_ret
    
    cumsum = np.empty(n_ret)
    cumsum[0] = returns[0] - mean_val
    for i in range(1, n_ret):
        cumsum[i] = cumsum[i-1] + (returns[i] - mean_val)
    
    R = np.max(cumsum) - np.min(cumsum)
    
    var_sum = 0.0
    for i in range(n_ret):
        var_sum += (returns[i] - mean_val) ** 2
    S = np.sqrt(var_sum / (n_ret - 1))
    
    if S > 1e-10 and R > 1e-10:
        RS = R / S
        H = np.log(RS) / np.log(n_ret)
        return min(max(H, 0.2), 0.8)
    
    return 0.5


@jit(nopython=True, cache=True)
def _hurst_multiscale_rs(prices: np.ndarray) -> float:
    """Calculate Hurst exponent using multi-scale R/S analysis."""
    n = len(prices)
    if n < 50:
        return 0.5
    
    returns = np.empty(n - 1)
    for i in range(n - 1):
        if prices[i] > 0 and prices[i + 1] > 0:
            returns[i] = np.log(prices[i + 1] / prices[i])
        else:
            returns[i] = 0.0
    
    n_ret = len(returns)
    if n_ret < 20:
        return 0.5
    
    min_period = 8
    max_period = n_ret // 4
    
    if max_period < min_period:
        return _hurst_rs_single_fixed(prices)
    
    num_periods = min(10, max_period - min_period + 1)
    if num_periods < 3:
        return _hurst_rs_single_fixed(prices)
    
    log_n = np.empty(num_periods)
    log_rs = np.empty(num_periods)
    valid_count = 0
    
    ratio = (max_period / min_period) ** (1.0 / (num_periods - 1))
    
    for i in range(num_periods):
        period = int(min_period * (ratio ** i))
        if period > max_period:
            period = max_period
        
        rs = _calculate_rs_for_period(returns, period)
        
        if rs > 0:
            log_n[valid_count] = np.log(period)
            log_rs[valid_count] = np.log(rs)
            valid_count += 1
    
    if valid_count < 3:
        return _hurst_rs_single_fixed(prices)
    
    log_n = log_n[:valid_count]
    log_rs = log_rs[:valid_count]
    
    mean_x = 0.0
    mean_y = 0.0
    for i in range(valid_count):
        mean_x += log_n[i]
        mean_y += log_rs[i]
    mean_x /= valid_count
    mean_y /= valid_count
    
    numerator = 0.0
    denominator = 0.0
    for i in range(valid_count):
        dx = log_n[i] - mean_x
        dy = log_rs[i] - mean_y
        numerator += dx * dy
        denominator += dx * dx
    
    if denominator < 1e-10:
        return 0.5
    
    H = numerator / denominator
    return min(max(H, 0.2), 0.8)


@jit(nopython=True, parallel=True, cache=True)
def calculate_hurst_vectorized(prices: np.ndarray, window: int = 100, 
                                use_multiscale: bool = True) -> np.ndarray:
    """Vectorized Hurst exponent using Numba parallel processing."""
    n = len(prices)
    hurst = np.full(n, 0.5)
    
    for i in prange(window, n):
        price_window = prices[i-window:i]
        if use_multiscale:
            hurst[i] = _hurst_multiscale_rs(price_window)
        else:
            hurst[i] = _hurst_rs_single_fixed(price_window)
    
    return hurst


@jit(nopython=True, cache=True)
def detect_fractals_fixed_n(highs: np.ndarray, lows: np.ndarray, 
                            n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect fractal pivots for a fixed n value."""
    length = len(highs)
    
    max_pivots = length // (n + 1) + 1
    pivot_high_bars = np.empty(max_pivots, dtype=np.int64)
    pivot_high_prices = np.empty(max_pivots, dtype=np.float64)
    pivot_low_bars = np.empty(max_pivots, dtype=np.int64)
    pivot_low_prices = np.empty(max_pivots, dtype=np.float64)
    
    ph_count = 0
    pl_count = 0
    
    for i in range(n, length - n):
        # Check for pivot high
        is_high = True
        for j in range(1, n + 1):
            if not (highs[i] > highs[i - j] and highs[i] > highs[i + j]):
                is_high = False
                break
        
        if is_high:
            confirm_bar = i + n
            if confirm_bar < length:
                pivot_high_bars[ph_count] = confirm_bar
                pivot_high_prices[ph_count] = highs[i]
                ph_count += 1
        
        # Check for pivot low
        is_low = True
        for j in range(1, n + 1):
            if not (lows[i] < lows[i - j] and lows[i] < lows[i + j]):
                is_low = False
                break
        
        if is_low:
            confirm_bar = i + n
            if confirm_bar < length:
                pivot_low_bars[pl_count] = confirm_bar
                pivot_low_prices[pl_count] = lows[i]
                pl_count += 1
    
    return (pivot_high_bars[:ph_count], pivot_high_prices[:ph_count],
            pivot_low_bars[:pl_count], pivot_low_prices[:pl_count])


@jit(nopython=True, cache=True)
def _kmeans_2_clusters_1d(prices: np.ndarray, max_iter: int = 10) -> Tuple[np.ndarray, float, float, int, int]:
    """Fast 2-means clustering for 1D data."""
    n = len(prices)
    labels = np.zeros(n, dtype=np.int32)
    
    if n < 2:
        return labels, prices[0] if n > 0 else 0.0, 0.0, n, 0
    
    c0 = np.min(prices)
    c1 = np.max(prices)
    
    if c0 == c1:
        return labels, c0, c1, n, 0
    
    count0, count1 = 0, 0
    
    for _ in range(max_iter):
        count0, count1 = 0, 0
        for i in range(n):
            if abs(prices[i] - c0) <= abs(prices[i] - c1):
                labels[i] = 0
                count0 += 1
            else:
                labels[i] = 1
                count1 += 1
        
        if count0 == 0 or count1 == 0:
            break
        
        sum0, sum1 = 0.0, 0.0
        for i in range(n):
            if labels[i] == 0:
                sum0 += prices[i]
            else:
                sum1 += prices[i]
        
        new_c0 = sum0 / count0
        new_c1 = sum1 / count1
        
        if abs(new_c0 - c0) < 1e-9 and abs(new_c1 - c1) < 1e-9:
            break
        
        c0, c1 = new_c0, new_c1
    
    return labels, c0, c1, count0, count1


@jit(nopython=True, cache=True)
def _cluster_trend_direction_numba(prices: np.ndarray, bars: np.ndarray) -> int:
    """Determine if clustered pivots are rising or falling."""
    n = len(prices)
    if n < 2:
        return 0
    
    labels, c0, c1, count0, count1 = _kmeans_2_clusters_1d(prices)
    
    if count0 == 0 or count1 == 0:
        return 0
    
    bar_sum0, bar_sum1 = 0.0, 0.0
    for i in range(n):
        if labels[i] == 0:
            bar_sum0 += bars[i]
        else:
            bar_sum1 += bars[i]
    
    avg_bar_0 = bar_sum0 / count0
    avg_bar_1 = bar_sum1 / count1
    
    if avg_bar_0 > avg_bar_1:
        newer_centroid = c0
        older_centroid = c1
    else:
        newer_centroid = c1
        older_centroid = c0
    
    if newer_centroid > older_centroid:
        return 1
    elif newer_centroid < older_centroid:
        return -1
    return 0


@jit(nopython=True, cache=True)
def _compute_trend_numba(
    length: int,
    adaptive_lookback: np.ndarray,
    ph_bars: np.ndarray,
    ph_prices: np.ndarray,
    pl_bars: np.ndarray,
    pl_prices: np.ndarray,
    min_pivots: int,
    trend_persistence: int
) -> np.ndarray:
    """Compute pivot cluster trend for each bar."""
    pivot_trend = np.zeros(length, dtype=np.int8)
    last_valid_trend = np.int8(0)
    bars_since_valid = 0
    
    ph_bars_float = ph_bars.astype(np.float64)
    pl_bars_float = pl_bars.astype(np.float64)
    
    for i in range(length):
        lookback = adaptive_lookback[i]
        window_start = max(0, i - lookback)
        
        high_start = np.searchsorted(ph_bars, window_start)
        high_end = np.searchsorted(ph_bars, i + 1)
        
        low_start = np.searchsorted(pl_bars, window_start)
        low_end = np.searchsorted(pl_bars, i + 1)
        
        n_highs = high_end - high_start
        n_lows = low_end - low_start
        
        if n_highs < min_pivots or n_lows < min_pivots:
            if last_valid_trend != 0 and bars_since_valid < trend_persistence:
                pivot_trend[i] = last_valid_trend
                bars_since_valid += 1
            continue
        
        window_high_prices = ph_prices[high_start:high_end]
        window_high_bars = ph_bars_float[high_start:high_end]
        window_low_prices = pl_prices[low_start:low_end]
        window_low_bars = pl_bars_float[low_start:low_end]
        
        highs_direction = _cluster_trend_direction_numba(window_high_prices, window_high_bars)
        lows_direction = _cluster_trend_direction_numba(window_low_prices, window_low_bars)
        
        if highs_direction != 0 and lows_direction != 0:
            if highs_direction == 1 and lows_direction == 1:
                pivot_trend[i] = np.int8(1)
                last_valid_trend = np.int8(1)
                bars_since_valid = 0
            elif highs_direction == -1 and lows_direction == -1:
                pivot_trend[i] = np.int8(-1)
                last_valid_trend = np.int8(-1)
                bars_since_valid = 0
            else:
                if last_valid_trend != 0 and bars_since_valid < trend_persistence:
                    pivot_trend[i] = last_valid_trend
                    bars_since_valid += 1
        else:
            if last_valid_trend != 0 and bars_since_valid < trend_persistence:
                pivot_trend[i] = last_valid_trend
                bars_since_valid += 1
    
    return pivot_trend


# =============================================================================
# PIVOT TREND CALCULATOR
# =============================================================================

class PivotTrendCalculator:
    """Calculate pivot cluster trend with adaptive parameters."""
    
    def __init__(self, config: Optional[PivotTrendConfig] = None):
        self.config = config or PivotTrendConfig()
        self._ph_bars = np.array([], dtype=np.int64)
        self._ph_prices = np.array([], dtype=np.float64)
        self._pl_bars = np.array([], dtype=np.int64)
        self._pl_prices = np.array([], dtype=np.float64)
    
    def calculate(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Calculate pivot cluster trend for OHLC data."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"CALCULATING PIVOT CLUSTER TREND")
            print(f"{'='*60}")
            print(f"Input: {len(df):,} bars")
        
        result = df.copy()
        cfg = self.config
        
        # Calculate adaptive parameters
        if verbose:
            print("Calculating adaptive parameters...")
        
        # ATR-based fractal N
        atr = ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, 
                     timeperiod=cfg.ATR_PERIOD)
        atr_pct = atr / df['Close'].values * 100
        
        atr_series = pd.Series(atr_pct, index=df.index)
        rolling_min = atr_series.rolling(window=cfg.ATR_NORM_WINDOW, min_periods=1).min()
        rolling_max = atr_series.rolling(window=cfg.ATR_NORM_WINDOW, min_periods=1).max()
        
        range_val = (rolling_max - rolling_min).replace(0, 1)
        atr_norm = ((atr_series - rolling_min) / range_val).fillna(0.5).clip(0, 1)
        
        fractal_n = cfg.MIN_FRACTAL_N + atr_norm * (cfg.MAX_FRACTAL_N - cfg.MIN_FRACTAL_N)
        result['adaptive_fractal_n'] = fractal_n.round().astype(int)
        
        # Hurst-based lookback
        prices = df['Close'].values
        hurst = calculate_hurst_vectorized(
            prices, 
            window=cfg.HURST_WINDOW,
            use_multiscale=cfg.HURST_USE_MULTISCALE
        )
        result['hurst_exponent'] = hurst
        
        hurst_norm = np.clip((hurst - 0.3) / 0.4, 0, 1)
        lookback = cfg.MAX_LOOKBACK - hurst_norm * (cfg.MAX_LOOKBACK - cfg.MIN_LOOKBACK)
        result['adaptive_lookback'] = lookback.round().astype(int)
        
        if verbose:
            print(f"  Fractal N range: {result['adaptive_fractal_n'].min()}-{result['adaptive_fractal_n'].max()}")
            print(f"  Hurst mean: {result['hurst_exponent'].mean():.3f}")
            print(f"  Lookback mean: {result['adaptive_lookback'].mean():.1f}")
        
        # Calculate pivot cluster trend
        if verbose:
            print("Detecting fractals and computing trend...")
        
        highs = df['High'].values.astype(np.float64)
        lows = df['Low'].values.astype(np.float64)
        adaptive_fractal_n = result['adaptive_fractal_n'].values.astype(np.int64)
        adaptive_lookback = result['adaptive_lookback'].values.astype(np.int64)
        length = len(df)
        
        # Pre-compute fractals for all unique N values
        fractal_cache = {}
        unique_n_values = np.unique(adaptive_fractal_n)
        
        for n in unique_n_values:
            ph_bars, ph_prices, pl_bars, pl_prices = detect_fractals_fixed_n(highs, lows, int(n))
            fractal_cache[n] = (ph_bars, ph_prices, pl_bars, pl_prices)
        
        # Build combined pivot arrays
        all_ph_bars_list = []
        all_ph_prices_list = []
        all_pl_bars_list = []
        all_pl_prices_list = []
        
        for n, (ph_bars, ph_prices, pl_bars, pl_prices) in fractal_cache.items():
            for i in range(len(ph_bars)):
                cb = ph_bars[i]
                if cb < length and adaptive_fractal_n[cb] == n:
                    all_ph_bars_list.append(cb)
                    all_ph_prices_list.append(ph_prices[i])
            
            for i in range(len(pl_bars)):
                cb = pl_bars[i]
                if cb < length and adaptive_fractal_n[cb] == n:
                    all_pl_bars_list.append(cb)
                    all_pl_prices_list.append(pl_prices[i])
        
        all_ph_bars = np.array(all_ph_bars_list, dtype=np.int64)
        all_ph_prices = np.array(all_ph_prices_list, dtype=np.float64)
        all_pl_bars = np.array(all_pl_bars_list, dtype=np.int64)
        all_pl_prices = np.array(all_pl_prices_list, dtype=np.float64)
        
        # Sort by bar index
        if len(all_ph_bars) > 0:
            ph_sort_idx = np.argsort(all_ph_bars)
            all_ph_bars = all_ph_bars[ph_sort_idx]
            all_ph_prices = all_ph_prices[ph_sort_idx]
        
        if len(all_pl_bars) > 0:
            pl_sort_idx = np.argsort(all_pl_bars)
            all_pl_bars = all_pl_bars[pl_sort_idx]
            all_pl_prices = all_pl_prices[pl_sort_idx]
        
        if verbose:
            print(f"  Found {len(all_ph_bars)} pivot highs, {len(all_pl_bars)} pivot lows")
        
        # Compute trend
        pivot_trend = _compute_trend_numba(
            length,
            adaptive_lookback,
            all_ph_bars,
            all_ph_prices,
            all_pl_bars,
            all_pl_prices,
            cfg.MIN_PIVOTS,
            cfg.TREND_PERSISTENCE
        )
        
        result['pivot_cluster_trend'] = pivot_trend
        result['pivot_bull'] = pivot_trend == 1
        result['pivot_bear'] = pivot_trend == -1
        
        # Store pivot locations for visualization
        self._ph_bars = all_ph_bars
        self._ph_prices = all_ph_prices
        self._pl_bars = all_pl_bars
        self._pl_prices = all_pl_prices
        
        if verbose:
            bull_bars = (pivot_trend == 1).sum()
            bear_bars = (pivot_trend == -1).sum()
            neutral_bars = (pivot_trend == 0).sum()
            total = len(pivot_trend)
            print(f"✓ Trend calculated:")
            print(f"  Bullish:  {bull_bars:,} bars ({100*bull_bars/total:.1f}%)")
            print(f"  Bearish:  {bear_bars:,} bars ({100*bear_bars/total:.1f}%)")
            print(f"  Neutral:  {neutral_bars:,} bars ({100*neutral_bars/total:.1f}%)")
        
        return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_pivot_trend_chart(
    df: pd.DataFrame,
    calculator: PivotTrendCalculator,
    symbol: str,
    timeframe: str,
    show_pivots: bool = True,
    show_hurst: bool = False,
    height: int = 800,
    display_start: str = None,
    display_end: str = None
) -> go.Figure:
    """Create interactive Plotly chart with OHLC and colored trend backgrounds."""
    
    # Store original df for pivot filtering
    original_df = df.copy()
    original_start_idx = 0
    
    # Apply display date range filter
    if display_start is not None or display_end is not None:
        mask = pd.Series([True] * len(df), index=df.index)
        
        if display_start is not None:
            start_dt = pd.to_datetime(display_start)
            mask &= (df.index >= start_dt)
            # Find the original index where display starts
            original_start_idx = df.index.get_indexer([start_dt], method='bfill')[0]
            if original_start_idx < 0:
                original_start_idx = 0
        
        if display_end is not None:
            end_dt = pd.to_datetime(display_end)
            mask &= (df.index <= end_dt)
        
        df = df[mask].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data in display range: {display_start} to {display_end}")
        
        print(f"Display range: {df.index[0]} to {df.index[-1]} ({len(df):,} bars)")
    
    # Create figure with subplots if showing Hurst
    if show_hurst:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2],
            subplot_titles=[None, 'Hurst Exponent']
        )
    else:
        fig = go.Figure()
    
    # Create x-axis with sequential indices to remove weekend gaps
    x_indices = list(range(len(df)))
    
    # Create hover text with actual dates
    hover_dates = df.index.strftime('%Y-%m-%d %H:%M')
    
    # Build hover text for candlesticks
    hover_text = [
        f"{date}<br>Open: ${o:.2f}<br>High: ${h:.2f}<br>Low: ${l:.2f}<br>Close: ${c:.2f}"
        for date, o, h, l, c in zip(
            hover_dates, 
            df['Open'], 
            df['High'], 
            df['Low'], 
            df['Close']
        )
    ]
    
    # Find contiguous trend zones for background coloring
    trend = df['pivot_cluster_trend'].values
    zones = []
    current_zone_start = 0
    current_trend = trend[0]
    
    for i in range(1, len(trend)):
        if trend[i] != current_trend:
            if current_trend != 0:
                zones.append((current_zone_start, i - 1, current_trend))
            current_zone_start = i
            current_trend = trend[i]
    
    # Add last zone
    if current_trend != 0:
        zones.append((current_zone_start, len(trend) - 1, current_trend))
    
    # Get price range for background rectangles
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_range = price_max - price_min
    y0 = price_min - price_range * 0.02
    y1 = price_max + price_range * 0.02
    
    # Add background zones as shapes
    shapes = []
    for zone_start, zone_end, zone_trend in zones:
        if zone_trend == 1:
            color = 'rgba(0, 200, 83, 0.12)'
        else:
            color = 'rgba(255, 82, 82, 0.12)'
        
        shapes.append(dict(
            type='rect',
            xref='x',
            yref='y' if not show_hurst else 'y1',
            x0=zone_start - 0.5,
            x1=zone_end + 0.5,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            layer='below'
        ))
    
    # Add candlestick chart
    candlestick = go.Candlestick(
        x=x_indices,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        text=hover_text,
        hoverinfo='text'
    )
    
    if show_hurst:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add 200 EMA line if available
    if 'ema_200' in df.columns:
        ema_trace = go.Scatter(
            x=x_indices,
            y=df['ema_200'],
            mode='lines',
            name='EMA 200',
            line=dict(color='#f6d55c', width=1.6),
            hoverinfo='skip'
        )
        if show_hurst:
            fig.add_trace(ema_trace, row=1, col=1)
        else:
            fig.add_trace(ema_trace)
    
    # Add pivot markers if requested
    if show_pivots and len(calculator._ph_bars) > 0:
        ph_bars = calculator._ph_bars
        ph_prices = calculator._ph_prices
        pl_bars = calculator._pl_bars
        pl_prices = calculator._pl_prices
        
        # Get the display range in terms of original indices
        display_start_original = original_start_idx
        display_end_original = original_start_idx + len(df)
        
        # Filter pivots to display range and adjust indices
        valid_ph = (ph_bars >= display_start_original) & (ph_bars < display_end_original)
        valid_pl = (pl_bars >= display_start_original) & (pl_bars < display_end_original)
        
        # Adjust pivot bar indices relative to display start
        ph_bars_adjusted = ph_bars[valid_ph] - display_start_original
        pl_bars_adjusted = pl_bars[valid_pl] - display_start_original
        ph_prices_filtered = ph_prices[valid_ph]
        pl_prices_filtered = pl_prices[valid_pl]
        
        # Build hover text for pivots
        ph_hover_text = [
            f"Pivot High<br>{hover_dates[b]}<br>Price: ${p:.2f}"
            for b, p in zip(ph_bars_adjusted, ph_prices_filtered)
        ]
        pl_hover_text = [
            f"Pivot Low<br>{hover_dates[b]}<br>Price: ${p:.2f}"
            for b, p in zip(pl_bars_adjusted, pl_prices_filtered)
        ]
        
        # Pivot highs
        if len(ph_bars_adjusted) > 0:
            pivot_high_trace = go.Scatter(
                x=ph_bars_adjusted.tolist(),
                y=(ph_prices_filtered * 1.002).tolist(),
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='#ff6b6b',
                    line=dict(color='darkred', width=1)
                ),
                name='Pivot High',
                text=ph_hover_text,
                hoverinfo='text'
            )
            
            if show_hurst:
                fig.add_trace(pivot_high_trace, row=1, col=1)
            else:
                fig.add_trace(pivot_high_trace)
        
        # Pivot lows
        if len(pl_bars_adjusted) > 0:
            pivot_low_trace = go.Scatter(
                x=pl_bars_adjusted.tolist(),
                y=(pl_prices_filtered * 0.998).tolist(),
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='#51cf66',
                    line=dict(color='darkgreen', width=1)
                ),
                name='Pivot Low',
                text=pl_hover_text,
                hoverinfo='text'
            )
            
            if show_hurst:
                fig.add_trace(pivot_low_trace, row=1, col=1)
            else:
                fig.add_trace(pivot_low_trace)
    
    # Add Hurst subplot if requested
    if show_hurst and 'hurst_exponent' in df.columns:
        # Build hover text for Hurst
        hurst_hover_text = [
            f"{date}<br>Hurst: {h:.3f}"
            for date, h in zip(hover_dates, df['hurst_exponent'])
        ]
        
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=df['hurst_exponent'],
            mode='lines',
            name='Hurst',
            line=dict(color='#9775fa', width=1.5),
            text=hurst_hover_text,
            hoverinfo='text'
        ), row=2, col=1)
        
        # Add reference lines
        fig.add_hline(y=0.5, line_dash='dash', line_color='rgba(255,255,255,0.3)', 
                      annotation_text='Random Walk (0.5)', 
                      annotation_position='right',
                      row=2, col=1)
        fig.add_hline(y=0.3, line_dash='dot', line_color='rgba(255,82,82,0.3)', row=2, col=1)
        fig.add_hline(y=0.7, line_dash='dot', line_color='rgba(0,200,83,0.3)', row=2, col=1)
        
        fig.update_yaxes(range=[0.2, 0.8], row=2, col=1)
    
    # Create tick labels
    n_ticks = min(25, len(df) // 10)
    if n_ticks < 1:
        n_ticks = 1
    tick_step = max(1, len(df) // n_ticks)
    tickvals = list(range(0, len(df), tick_step))
    ticktext = [df.index[i].strftime('%m/%d\n%H:%M') for i in tickvals]
    
    # Build title with date range info
    date_range_str = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
    title_text = f'<b>{symbol}</b> - {timeframe} | Pivot Cluster Trend<br><sup>{date_range_str}</sup>'
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18),
            x=0.5
        ),
        xaxis=dict(
            title='',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            showspikes=True,
            spikecolor='gray',
            spikethickness=1
        ),
        yaxis=dict(
            title='Price ($)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            tickformat=',.2f'
        ),
        shapes=shapes,
        height=height,
        template='plotly_dark',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        margin=dict(l=70, r=50, t=100, b=70),
        paper_bgcolor='rgb(17, 17, 17)',
        plot_bgcolor='rgb(17, 17, 17)'
    )
    
    # Add legend annotation
    bull_count = (trend == 1).sum()
    bear_count = (trend == -1).sum()
    neutral_count = (trend == 0).sum()
    
    annotation_text = (
        f'<b>Trend Legend:</b>  '
        f'<span style="color:#00c853">■ Bullish ({bull_count:,} bars)</span>  |  '
        f'<span style="color:#ff5252">■ Bearish ({bear_count:,} bars)</span>  |  '
        f'Neutral ({neutral_count:,} bars)'
    )
    
    fig.add_annotation(
        x=0.5, y=1.08,
        xref='paper', yref='paper',
        text=annotation_text,
        showarrow=False,
        font=dict(size=11, color='white'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='rgba(128,128,128,0.5)',
        borderwidth=1,
        borderpad=6
    )
    
    return fig


def visualize_pivot_trend(
    symbol: str,
    display_timeframe: str,
    base_timeframe: str = "1min",
    years_back: float = 2.0,
    show_pivots: bool = True,
    show_hurst: bool = False,
    chart_height: int = 800,
    use_cache: bool = True,
    save_html: bool = False,
    output_dir: Optional[Path] = None,
    display_start: str = None,
    display_end: str = None,
    verbose: bool = True
) -> go.Figure:
    """
    Main visualization function - fetches data, calculates trend, and creates chart.
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PIVOT CLUSTER TREND VISUALIZATION")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Display Timeframe: {display_timeframe}")
        print(f"Base Timeframe: {base_timeframe}")
        print(f"Years Back: {years_back}")
        if display_start or display_end:
            print(f"Display Range: {display_start or 'start'} to {display_end or 'end'}")
    
    # Step 1: Fetch data
    if verbose:
        print(f"\nFetching {base_timeframe} data...")
    
    df_base = fetch_intraday_data(
        symbol=symbol,
        timeframe=base_timeframe,
        years_back=years_back,
        use_cache=use_cache
    )
    
    if verbose:
        print(f"  Fetched {len(df_base):,} bars from {df_base.index[0]} to {df_base.index[-1]}")
    
    # Step 2: Resample to display timeframe
    if verbose:
        print(f"\nResampling to {display_timeframe}...")
    
    df = resample_to_timeframe(df_base, display_timeframe)
    
    # Compute 200 EMA on the display timeframe
    df['ema_200'] = ta.EMA(df['Close'].values, timeperiod=200)
    
    if verbose:
        print(f"  Resampled to {len(df):,} bars")
    
    # Step 3: Calculate pivot cluster trend
    calculator = PivotTrendCalculator()
    result_df = calculator.calculate(df, verbose=verbose)
    
    # Step 4: Create chart
    if verbose:
        print(f"\nCreating interactive chart...")
    
    fig = create_pivot_trend_chart(
        df=result_df,
        calculator=calculator,
        symbol=symbol,
        timeframe=display_timeframe,
        show_pivots=show_pivots,
        show_hurst=show_hurst,
        height=chart_height,
        display_start=display_start,
        display_end=display_end
    )
    
    # Step 5: Save HTML if requested
    if save_html:
        if output_dir is None:
            output_dir = Path("./charts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{display_timeframe}_pivot_trend_{timestamp}.html"
        filepath = output_dir / filename
        
        fig.write_html(str(filepath))
        if verbose:
            print(f"\n✓ Chart saved to: {filepath}")
    
    if verbose:
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*60}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    """Run visualization with parameters from CONFIG section at top of file."""
    
    fig = visualize_pivot_trend(
        symbol=SYMBOL,
        display_timeframe=DISPLAY_TIMEFRAME,
        base_timeframe=BASE_TIMEFRAME,
        years_back=YEARS_BACK,
        show_pivots=SHOW_PIVOTS,
        show_hurst=SHOW_HURST,
        chart_height=CHART_HEIGHT,
        use_cache=USE_CACHE,
        save_html=SAVE_HTML,
        output_dir=Path(OUTPUT_DIR) if OUTPUT_DIR else None,
        display_start=DISPLAY_START_DATE,
        display_end=DISPLAY_END_DATE,
        verbose=True
    )
    
    # Show the figure in browser
    fig.show()

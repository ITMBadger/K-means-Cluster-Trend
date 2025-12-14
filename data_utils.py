# data_utils.py
"""
Data fetching, caching, and utility functions for trading strategy.

Pipeline for intraday data (fetch_intraday_data):

1. Fetch all raw OHLCV from Tiingo IEX intraday endpoint.
2. Use pandas_market_calendars (NYSE) to:
   - Restrict to the official market_open → market_close interval per day
   - Remove holidays and non-session days completely
   - Keep early-close days, but only up to their official early close time
     (TradingView-style behavior).
3. Call Tiingo daily prices endpoint, read splitFactor, and
   backward-adjust intraday OHLC for stock splits.
"""

import os
import numpy as np
import pandas as pd
import requests
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import time
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    ZoneInfo = pytz.timezone


# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
env_path = SCRIPT_DIR / '.env'
load_dotenv(dotenv_path=env_path, override=False)

API_TOKEN = os.getenv("TIINGO_API_TOKEN")
NY_TZ = ZoneInfo("America/New_York")

BACKTEST_OUTPUT_DIR = SCRIPT_DIR / "backtest_results"
DATA_CACHE_DIR = SCRIPT_DIR / "data_cache"


# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def get_cache_filename(symbol: str, timeframe: str,
                       start_date: datetime, end_date: datetime,
                       is_ipo: bool = False) -> str:
    """Generate cache filename from parameters."""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    if is_ipo:
        return f"{symbol}_{timeframe}_IPO_{start_str}_{end_str}.csv"
    return f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"


def _validate_cached_data(cached_df: pd.DataFrame, start_date: datetime,
                          tolerance_days: int = 7, verbose: bool = True,
                          is_ipo: bool = False) -> bool:
    """Check if cached data has sufficient historical coverage."""
    if len(cached_df) == 0:
        return False

    # For IPO stocks, don't validate start date
    if is_ipo:
        if verbose:
            print(f"   ✅ Cache valid (IPO): {len(cached_df):,} bars")
        return True

    # For regular stocks, validate start date
    cached_start = cached_df.index.min()
    tolerance = timedelta(days=tolerance_days)

    if cached_start <= start_date + tolerance:
        if verbose:
            print(f"   ✅ Cache valid: {len(cached_df):,} bars")
        return True
    return False


def _try_load_cache(filepath: Path, start_date: datetime,
                    verbose: bool = True, is_ipo: bool = False) -> Optional[pd.DataFrame]:
    """Try to load and validate a cache file."""
    try:
        if verbose:
            print(f"📂 Cache found: {filepath.name}")
        cached_df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        if _validate_cached_data(cached_df, start_date, verbose=verbose, is_ipo=is_ipo):
            return cached_df
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Cache error: {e}")
    return None


def check_cache_exists(symbol: str, timeframe: str, years_back: int,
                       cache_dir: Path = None, verbose: bool = True) -> tuple:
    """Check for existing cache, first exact match then within tolerance."""
    if cache_dir is None:
        cache_dir = DATA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    # Try exact match first (non-IPO)
    cache_filename = get_cache_filename(symbol, timeframe, start_date, end_date, is_ipo=False)
    cache_filepath = cache_dir / cache_filename

    if cache_filepath.exists():
        cached_df = _try_load_cache(cache_filepath, start_date, verbose=verbose, is_ipo=False)
        if cached_df is not None:
            return True, cache_filepath, cached_df, start_date, end_date

    # Look for cache files within tolerance (both regular and IPO)
    pattern = f"{symbol}_{timeframe}_*.csv"
    candidate_files = list(cache_dir.glob(pattern))

    for candidate in candidate_files:
        try:
            parts = candidate.name.replace('.csv', '').split('_')

            # Check if this is an IPO file
            is_ipo_file = 'IPO' in candidate.name

            if is_ipo_file:
                # Format: {symbol}_{timeframe}_IPO_{start}_{end}.csv
                if len(parts) != 5:
                    continue
                cand_symbol, cand_timeframe, ipo_marker, cand_start_str, cand_end_str = parts

                if ipo_marker != 'IPO':
                    continue
            else:
                # Format: {symbol}_{timeframe}_{start}_{end}.csv
                if len(parts) != 4:
                    continue
                cand_symbol, cand_timeframe, cand_start_str, cand_end_str = parts

            # Check symbol and timeframe match
            if cand_symbol != symbol or cand_timeframe != timeframe:
                continue

            # Parse dates
            cand_start = datetime.strptime(cand_start_str, "%Y%m%d")
            cand_end = datetime.strptime(cand_end_str, "%Y%m%d")

            # Check end date tolerance
            end_diff = abs((cand_end - end_date).days)

            if is_ipo_file:
                # For IPO files, only check end date tolerance
                if end_diff <= 7:
                    cached_df = _try_load_cache(candidate, start_date, verbose=verbose, is_ipo=True)
                    if cached_df is not None:
                        return True, candidate, cached_df, start_date, end_date
            else:
                # For regular files, check both start and end dates
                start_is_sufficient = cand_start <= start_date + timedelta(days=7)
                if end_diff <= 7 and start_is_sufficient:
                    cached_df = _try_load_cache(candidate, start_date, verbose=verbose, is_ipo=False)
                    if cached_df is not None:
                        return True, candidate, cached_df, start_date, end_date

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Skipping {candidate.name}: {e}")
            continue

    if verbose:
        print(f"📂 Cache miss: No suitable cache found")
    return False, None, None, start_date, end_date


def save_to_cache(df: pd.DataFrame, symbol: str, timeframe: str,
                  target_start_date: datetime, target_end_date: datetime,
                  cache_dir: Path = None, verbose: bool = True) -> str:
    """
    Save DataFrame to cache.

    - Detects IPO (short history) and names file accordingly.
    - Rounds all float columns to 3 decimal places before writing CSV.
    """
    if cache_dir is None:
        cache_dir = DATA_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is IPO data (actual data doesn't go back to requested start date)
    actual_start_date = df.index.min()
    actual_end_date = df.index.max()

    # Convert to datetime if needed
    if hasattr(actual_start_date, 'to_pydatetime'):
        actual_start_date = actual_start_date.to_pydatetime()
    if hasattr(actual_end_date, 'to_pydatetime'):
        actual_end_date = actual_end_date.to_pydatetime()

    # Check if data starts significantly later than requested (more than 30 days tolerance)
    # This indicates an IPO stock with limited history
    days_difference = (actual_start_date - target_start_date).days
    is_ipo = days_difference > 30

    if is_ipo:
        # Use actual date range for IPO stocks
        cache_filename = get_cache_filename(
            symbol, timeframe, actual_start_date, actual_end_date, is_ipo=True
        )
        if verbose:
            print(f"💾 Detected IPO/new stock - using actual date range")
    else:
        # Use target date range for stocks with full history
        cache_filename = get_cache_filename(
            symbol, timeframe, target_start_date, target_end_date, is_ipo=False
        )

    cache_filepath = cache_dir / cache_filename

    # Round float columns to 3 decimal places
    df_to_save = df.copy()
    float_cols = df_to_save.select_dtypes(include=["float64", "float32", "float"]).columns
    if len(float_cols) > 0:
        df_to_save[float_cols] = df_to_save[float_cols].round(3)

    df_to_save.to_csv(cache_filepath)

    if verbose:
        print(f"💾 Saved to cache: {cache_filename} ({len(df):,} bars)")

    return str(cache_filepath)


# =============================================================================
# DATA FETCHING & CORPORATE-ACTION ADJUSTMENT FUNCTIONS
# =============================================================================

def filter_to_market_hours(df: pd.DataFrame,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Filter intraday data to official NYSE market hours only.

    Behavior (TradingView-style):
      * Uses pandas_market_calendars NYSE schedule for the date range.
      * For each trading day:
          - Keep bars from that day's `market_open` to `market_close`
            (inclusive).
          - This automatically handles early-close days because the calendar's
            `market_close` is earlier on those days.
      * Any bars that fall on days NOT in the NYSE schedule (holidays,
        weekends, etc.) are dropped completely, which removes Tiingo's
        synthetic "flat price" holiday bars.

    Args:
        df: Intraday OHLCV DataFrame indexed by naive datetimes in ET.
        verbose: Print progress messages.

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    if verbose:
        print(f"Filtering to NYSE market hours...")

    start_date_str = df.index.min().strftime('%Y-%m-%d')
    end_date_str = df.index.max().strftime('%Y-%m-%d')

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date_str, end_date=end_date_str)

    if schedule.empty:
        if verbose:
            print("   ⚠️  NYSE schedule is empty for given range; returning original data.")
        return df

    # Convert schedule to naive ET
    schedule_et = schedule.copy()
    schedule_et['market_open'] = (
        schedule_et['market_open']
        .dt.tz_convert('America/New_York')
        .dt.tz_localize(None)
    )
    schedule_et['market_close'] = (
        schedule_et['market_close']
        .dt.tz_convert('America/New_York')
        .dt.tz_localize(None)
    )

    # Build mask of all valid intraday bars:
    # For each session date, allow only [market_open, market_close].
    valid_mask = pd.Series(False, index=df.index)
    for _, row in schedule_et.iterrows():
        day_mask = (df.index >= row['market_open']) & (df.index <= row['market_close'])
        valid_mask = valid_mask | day_mask

    filtered_df = df[valid_mask].copy()

    if verbose:
        print(f"✓ Filtered to NYSE market sessions: {len(df):,} → {len(filtered_df):,} bars")

    return filtered_df


def fetch_split_events(symbol: str,
                       start_date: datetime,
                       end_date: datetime,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Fetch stock split events for `symbol` from Tiingo daily prices endpoint.

    Uses the `splitFactor` field from:
        https://api.tiingo.com/tiingo/daily/{symbol}/prices

    Args:
        symbol: Ticker symbol.
        start_date: Earliest calendar date to request (inclusive).
        end_date: Latest calendar date to request (inclusive).
        verbose: Print diagnostics.

    Returns:
        DataFrame with columns:
            - 'date' (datetime.date): trading date of the split
            - 'splitFactor' (float): total split factor on that date
        Only rows with effective splitFactor != 1 are returned.
    """
    if not API_TOKEN:
        raise ValueError("TIINGO_API_TOKEN must be set in .env file")

    if verbose:
        print(f"\n   🔎 Fetching split history for {symbol} from Tiingo daily API...")

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {API_TOKEN}",
    }
    params = {
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Split API Error: {response.status_code} - {response.text}")

    data = response.json()
    if not data:
        if verbose:
            print("   ℹ️  No daily data returned for split lookup.")
        return pd.DataFrame(columns=["date", "splitFactor"])

    daily_df = pd.DataFrame(data)

    if "splitFactor" not in daily_df.columns:
        if verbose:
            print("   ℹ️  splitFactor not present in daily response; corporate actions may not be enabled on your plan.")
        return pd.DataFrame(columns=["date", "splitFactor"])

    # Convert date strings to pure calendar dates (ignore timezone)
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df["splitFactor"] = pd.to_numeric(daily_df["splitFactor"], errors="coerce")

    # Identify days where a split (or similar action) occurred
    mask = daily_df["splitFactor"].notna() & (np.abs(daily_df["splitFactor"] - 1.0) > 1e-6)
    split_events = daily_df.loc[mask, ["date", "splitFactor"]].copy()

    # If there are multiple corporate actions on the same date, combine them
    if not split_events.empty:
        split_events = (
            split_events
            .groupby("date", as_index=False)["splitFactor"]
            .prod()
            .sort_values("date")
        )

    if verbose:
        if split_events.empty:
            print("   ℹ️  No stock splits found in requested period.")
        else:
            first = split_events['date'].min()
            last = split_events['date'].max()
            print(f"   📌 Found {len(split_events)} split event(s) between {first} and {last}")

    return split_events


def apply_stock_splits(df: pd.DataFrame,
                       symbol: str,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Adjust intraday OHLC prices for historical stock splits using Tiingo daily data.

    Logic:
      * Fetch daily `splitFactor` between the first and last dates in `df`.
      * For each split date D with factor F:
          - Treat the split as effective at 09:30 ET on date D.
          - Divide all OHLC prices for timestamps strictly before that 09:30
            by F (backward adjustment).
        This ensures the most recent prices are unchanged and older prices
        are scaled to be on the same share-count basis.

    Notes:
      * Volume is left UNADJUSTED by default; intraday share counts remain actual.
      * Handles both forward splits (F > 1) and reverse splits (F < 1).

    Args:
        df: Intraday OHLCV DataFrame indexed by naive datetimes in ET.
        symbol: Ticker being adjusted (for API call).
        verbose: Print diagnostics.

    Returns:
        New DataFrame with split-adjusted OHLC prices.
    """
    if df.empty:
        return df

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if not price_cols:
        if verbose:
            print("   ⚠️  No OHLC columns found; skipping split adjustment.")
        return df

    first_dt = df.index.min()
    last_dt = df.index.max()
    start_date = first_dt.date()
    end_date = last_dt.date()

    try:
        split_events = fetch_split_events(
            symbol=symbol,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Unable to fetch split data: {e}")
            print("      Proceeding with UNADJUSTED intraday prices.")
        return df

    if split_events.empty:
        # Nothing to adjust
        return df

    adjusted = df.copy()

    for _, row in split_events.iterrows():
        split_date = row["date"]      # datetime.date
        factor = float(row["splitFactor"])

        if factor == 0 or np.isnan(factor):
            continue

        # Assume split effective at regular session open on that date (09:30 ET).
        split_dt = datetime.combine(split_date, datetime.min.time()).replace(
            hour=9, minute=30
        )

        # All bars strictly before the split open are scaled.
        mask = adjusted.index < split_dt
        affected = int(mask.sum())

        if affected == 0:
            continue

        if verbose:
            direction = "forward" if factor > 1.0 else "reverse"
            print(
                f"   🔧 Applying {direction} split (factor={factor}) "
                f"effective {split_date} 09:30 ET → adjusting {affected:,} bars"
            )

        adjusted.loc[mask, price_cols] = adjusted.loc[mask, price_cols] / factor

    if verbose:
        print("✓ Prices adjusted for historical stock splits")

    return adjusted


def fetch_intraday_data(symbol: str,
                        timeframe: str = "5min",
                        years_back: int = 2,
                        use_cache: bool = True,
                        cache_dir: Path = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Fetch intraday data from Tiingo IEX API with caching and stock-split adjustment.

    Processing order:
        1. Loop and fetch all raw OHLCV from Tiingo IEX endpoint.
        2. Use pandas_market_calendars (NYSE) to:
           - Restrict to each day's official `market_open` → `market_close`
             (includes early-close days with their shorter close).
           - Remove holidays and non-session days completely.
        3. Fetch stock split data (splitFactor) from Tiingo daily API and
           adjust intraday OHLC prices for splits.

    Args:
        symbol: Stock ticker symbol.
        timeframe: Data resolution (e.g., "1min", "5min", "15min").
        years_back: Number of years of historical data.
        use_cache: Whether to use cached data.
        cache_dir: Directory for cache files.
        verbose: Print progress messages.

    Returns:
        DataFrame with split-adjusted OHLCV data indexed by datetime (ET, naive).
    """
    if cache_dir is None:
        cache_dir = DATA_CACHE_DIR

    # -------------------------------------------------------------------------
    # 0. Try cache
    # -------------------------------------------------------------------------
    if use_cache:
        cache_exists, _, cached_data, target_start, target_end = check_cache_exists(
            symbol, timeframe, years_back, cache_dir, verbose
        )
        if cache_exists and cached_data is not None:
            return cached_data
    else:
        target_end = datetime.now()
        target_start = target_end - timedelta(days=years_back * 365)

    if not API_TOKEN:
        raise ValueError("TIINGO_API_TOKEN must be set in .env file")

    if verbose:
        print(f"\n🌐 Fetching {symbol} {timeframe} from Tiingo IEX API ({years_back} years)...")

    # -------------------------------------------------------------------------
    # 1. Loop and get all raw OHLCV from Tiingo IEX
    # -------------------------------------------------------------------------
    all_dataframes = []
    # `paging_end_date` used to walk backwards in time through IEX data
    paging_end_date: Optional[datetime] = None

    while True:
        url = f"https://api.tiingo.com/iex/{symbol}/prices"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {API_TOKEN}",
        }
        params = {
            "startDate": target_start.strftime("%Y-%m-%d"),
            "resampleFreq": timeframe,
            "columns": "open,high,low,close,volume",
        }
        if paging_end_date is not None:
            params["endDate"] = paging_end_date.strftime("%Y-%m-%d")

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"IEX API Error: {response.status_code} - {response.text}")

        data = response.json()
        if not data:
            break

        df = pd.DataFrame(data)
        # Tiingo IEX dates are in UTC; convert to US/Eastern, then drop tz
        df["datetime_utc"] = pd.to_datetime(df["date"], utc=True)
        df["time_et"] = df["datetime_utc"].dt.tz_convert("US/Eastern")

        df_chunk = df[["time_et", "open", "high", "low", "close", "volume"]].copy()
        df_chunk.set_index("time_et", inplace=True)
        df_chunk = df_chunk.sort_index()
        all_dataframes.append(df_chunk)

        earliest_utc = df["datetime_utc"].min()
        # Tiingo IEX returns at most 10,000 rows per call; if we got fewer or
        # reached target_start, we are done.
        if earliest_utc.replace(tzinfo=None) <= target_start or len(data) < 10000:
            break

        # Walk paging_end_date backward in time to continue fetching older data
        paging_end_date = earliest_utc.replace(tzinfo=None) + timedelta(days=1)
        time.sleep(0.5)  # Be gentle with the API

    if not all_dataframes:
        raise Exception("No intraday data retrieved from Tiingo IEX API")

    df_combined = pd.concat(all_dataframes)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined = df_combined.sort_index()

    # Make index naive ET (no timezone information)
    df_combined.index = df_combined.index.tz_localize(None)

    df_combined = df_combined.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    if verbose:
        print(f"✓ Fetched {len(df_combined):,} raw intraday bars")

    # -------------------------------------------------------------------------
    # 2. Restrict to NYSE regular market hours per calendar (TradingView-style)
    # -------------------------------------------------------------------------
    df_combined = filter_to_market_hours(
        df_combined,
        verbose=verbose,
    )

    # -------------------------------------------------------------------------
    # 3. Fetch split data from Tiingo daily API and adjust prices
    # -------------------------------------------------------------------------
    df_combined = apply_stock_splits(df_combined, symbol=symbol, verbose=verbose)

    # -------------------------------------------------------------------------
    # 4. Cache adjusted data (post-corporate-action) if requested
    # -------------------------------------------------------------------------
    if use_cache:
        save_to_cache(
            df_combined,
            symbol,
            timeframe,
            target_start,
            target_end,
            cache_dir,
            verbose,
        )

    return df_combined


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def resample_to_timeframe(df: pd.DataFrame, target_timeframe: str,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        df: DataFrame with OHLCV data.
        target_timeframe: Target resolution (e.g., "5min", "15min", "1hour").
        verbose: Print progress messages.

    Returns:
        Resampled DataFrame.
    """
    if verbose:
        print(f"Resampling to {target_timeframe}...")

    import re
    match = re.match(r'(\d+)(min|hour|h)$', target_timeframe.lower())
    if not match:
        raise ValueError(f"Invalid timeframe: {target_timeframe}")

    number = int(match.group(1))
    unit = match.group(2)
    freq = f"{number}min" if unit == 'min' else f"{number}h"

    first_date = df.index[0].date()
    anchor = pd.Timestamp(first_date).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    if df.index[0] < anchor:
        anchor = anchor - pd.Timedelta(days=1)

    resampled = (
        df.resample(freq, origin=anchor)
        .agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        })
        .dropna()
    )

    if verbose:
        print(f"✓ Resampled: {len(df):,} → {len(resampled):,} bars")

    return resampled


def map_signals_to_high_res(signals_df: pd.DataFrame,
                            high_res_data: pd.DataFrame,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Map strategy signals from lower resolution to high resolution timestamps.

    Uses binary search (np.searchsorted) for O(n log m) performance instead of
    O(n * m) with the naive loop approach.

    Args:
        signals_df: DataFrame with signals from strategy timeframe.
        high_res_data: DataFrame with high-resolution market data.
        verbose: Print progress messages.

    Returns:
        Signals DataFrame with mapped timestamps.
    """
    if signals_df.empty:
        return signals_df

    if verbose:
        print(f"\nMapping signals to high-resolution timestamps...")

    import time as timing
    start = timing.time()

    mapped_signals = signals_df.copy()
    high_res_index = high_res_data.index

    # Convert to numpy arrays for fast operations
    signal_times = pd.to_datetime(mapped_signals['DateTime_ET']).values
    high_res_times = high_res_index.values

    # Use searchsorted for O(log n) binary search per signal
    # 'side=left' finds first index where high_res_times >= signal_time
    indices = np.searchsorted(high_res_times, signal_times, side='left')

    # Clip indices to valid range (handle signals after last bar)
    indices = np.clip(indices, 0, len(high_res_times) - 1)

    # Map to actual timestamps
    mapped_times = high_res_times[indices]

    mapped_signals['DateTime_ET'] = mapped_times

    elapsed = timing.time() - start

    if verbose:
        print(f"✓ Mapped {len(signals_df)} signals to high-res timestamps ({elapsed*1000:.1f}ms)")

    return mapped_signals


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_signals(signals_df: pd.DataFrame, symbol: str, timeframe: str,
                   output_dir: Path = None) -> str:
    """
    Export signals to CSV file.

    Args:
        signals_df: DataFrame with signals.
        symbol: Stock ticker symbol.
        timeframe: Strategy timeframe.
        output_dir: Output directory for CSV file.

    Returns:
        Path to exported file as a string.
    """
    if output_dir is None:
        output_dir = BACKTEST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timeframe}_signals_{timestamp}.csv"
    filepath = output_dir / filename

    export_df = signals_df.copy()
    export_df['DateTime_ET'] = export_df['DateTime_ET'].astype(str)
    if 'Signal_Bar_DT' in export_df.columns:
        export_df['Signal_Bar_DT'] = export_df['Signal_Bar_DT'].astype(str)
    export_df.to_csv(filepath, index=False)

    print(f"✓ Signals saved: {filepath}")
    return str(filepath)

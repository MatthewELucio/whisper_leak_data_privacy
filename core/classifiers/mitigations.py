# core/mitigations.py

import abc
import math
import random
import numpy as np
import pandas as pd
import ast
from core.utils import PrintUtils # Assuming PrintUtils handles logging/printing

class BaseMitigation(abc.ABC):
    """
    Abstract base class for all mitigation strategies.

    Mitigations transform the input DataFrame (containing 'time_diffs'
    and 'data_lengths' columns) before normalization and model training.
    """

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the mitigation logic to the DataFrame.

        Args:
            df: The input DataFrame with 'time_diffs' and 'data_lengths' columns.
                These columns should contain lists of numbers.

        Returns:
            A new DataFrame with the mitigation applied.
        """
        pass

    def _ensure_list_format(self, series: pd.Series) -> pd.Series:
        """
        Ensures that the elements in the series are lists, attempting conversion
        from string representation if necessary.

        Args:
            series: A pandas Series, potentially containing string representations
                    of lists.

        Returns:
            A pandas Series where elements are lists.
        """
        # Check if the first element is already a list and non-empty
        first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
        if first_valid is not None and isinstance(first_valid, list):
            return series

        # Attempt conversion using ast.literal_eval
        PrintUtils.print_extra(f"Attempting list conversion for column '{series.name}'")
        try:
            return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
        except (ValueError, SyntaxError, TypeError) as e:
            PrintUtils.print_warning(f"Could not parse column '{series.name}' elements as lists: {e}. Mitigation might fail or be skipped.")
            # Return original but ensure elements are lists (even if empty on failure)
            return series.apply(lambda x: x if isinstance(x, list) else [])


class DataPaddingMitigation(BaseMitigation):
    """
    Mitigation that increases packet sizes to the nearest multiple of a block size.
    """
    def __init__(self, block_size: int = 16):
        """
        Initializes the data padding mitigation.

        Args:
            block_size: The target block size. Sizes will be padded up to the
                        nearest multiple of this value. Defaults to 16.
        """
        if block_size <= 0:
            raise ValueError("Block size must be positive.")
        self.block_size = block_size
        PrintUtils.print_extra(f"Initialized DataPaddingMitigation with block_size={self.block_size}")

    def _pad_sizes(self, sizes: list) -> list:
        """Pads a list of sizes."""
        if not isinstance(sizes, list): return sizes # Handle potential non-list data
        padded_sizes = []
        for s in sizes:
            # Ensure s is a number, default to 0 if not
            s_num = s if isinstance(s, (int, float)) else 0
            if s_num <= 0:
                padded_sizes.append(0) # Keep 0 or negative sizes as is
            else:
                padded_size = math.ceil(s_num / self.block_size) * self.block_size
                padded_sizes.append(int(padded_size)) # Store as int
        return padded_sizes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data padding to the 'data_lengths' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with 'data_lengths' padded.
        """
        PrintUtils.start_stage(f"Applying DataPaddingMitigation (block_size={self.block_size})")
        df_copy = df.copy()
        if 'data_lengths' not in df_copy.columns:
            PrintUtils.print_warning("Column 'data_lengths' not found. Skipping DataPaddingMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure data_lengths are lists
        df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])

        # Apply padding
        df_copy['data_lengths'] = df_copy['data_lengths'].apply(self._pad_sizes)
        PrintUtils.end_stage()
        return df_copy


class TimeNoiseMitigation(BaseMitigation):
    """
    Mitigation that adds random noise to non-zero inter-packet times.

    The noise added is a random percentage (between 0% and X%) of the
    median non-zero time difference observed in the dataset.
    """
    def __init__(self, noise_percentage: float = 0.10):
        """
        Initializes the time noise mitigation.

        Args:
            noise_percentage: The maximum percentage of the median non-zero time
                              difference to add as noise (e.g., 0.10 for 10%).
                              Defaults to 0.10.
        """
        if not 0.0 <= noise_percentage <= 1.0:
            raise ValueError("Noise percentage must be between 0.0 and 1.0.")
        self.noise_percentage = noise_percentage
        self.median_nonzero_time = None
        PrintUtils.print_extra(f"Initialized TimeNoiseMitigation with noise_percentage={self.noise_percentage*100:.1f}%")

    def _calculate_median_nonzero_time(self, series: pd.Series) -> float:
        """Calculates the median of all non-zero values across all lists in the series."""
        try:
            all_times = [time for sublist in series if isinstance(sublist, list) for time in sublist if isinstance(time, (int, float))]
            non_zero_times = [time for time in all_times if time > 1e-9] # Use small threshold for float comparison

            if not non_zero_times:
                return 0.0  # No non-zero times found

            return float(np.median(non_zero_times))
        except Exception as e:
            PrintUtils.print_warning(f"Could not calculate median time: {e}")
            return 0.0

    def _add_noise(self, times: list) -> list:
        """Adds noise to a list of times."""
        if not isinstance(times, list): return times # Handle potential non-list data
        if self.median_nonzero_time is None or self.median_nonzero_time <= 1e-9:
            return times # Cannot add noise if median is unknown or zero

        noisy_times = []
        max_noise = self.noise_percentage * self.median_nonzero_time
        for t in times:
            t_num = t if isinstance(t, (int, float)) else 0.0
            if t_num > 1e-9:
                noise = random.uniform(0, max_noise)
                noisy_times.append(t_num + noise)
            else:
                noisy_times.append(t_num) # Keep zero or negative times as is
        return noisy_times

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies time noise to the 'time_diffs' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with noise added to 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying TimeNoiseMitigation (noise={self.noise_percentage*100:.1f}%)")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_warning("Column 'time_diffs' not found. Skipping TimeNoiseMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        # Calculate median non-zero time difference across the dataset
        self.median_nonzero_time = self._calculate_median_nonzero_time(df_copy['time_diffs'])
        PrintUtils.print_extra(f"Median non-zero time difference: {self.median_nonzero_time:.6f}")

        if self.median_nonzero_time is None or self.median_nonzero_time <= 1e-9:
            PrintUtils.print_warning("Median non-zero time is zero or could not be calculated. Skipping noise addition.")
        else:
            # Apply noise
            df_copy['time_diffs'] = df_copy['time_diffs'].apply(self._add_noise)

        PrintUtils.end_stage()
        return df_copy


class LeakyBucketMitigation(BaseMitigation):
    """
    Mitigation that reshapes timing based on a leaky bucket / token bucket model.

    Simulates a token bucket where packets can only be "sent" if a token is
    available. Tokens regenerate at a constant rate. This introduces delays
    if packets arrive faster than the token regeneration rate, smoothing bursts.
    """
    def __init__(self, rate: float = 50.0, burst_size: float = 10.0):
        """
        Initializes the leaky bucket mitigation.

        Args:
            rate: The rate at which tokens regenerate (packets per second).
                  Defaults to 50.0.
            burst_size: The maximum number of tokens the bucket can hold
                        (maximum burst size in packets). Defaults to 10.0.
        """
        if rate <= 0:
            raise ValueError("Rate must be positive.")
        if burst_size < 1:
            raise ValueError("Burst size must be at least 1.")
        self.rate = rate
        self.burst_size = burst_size
        PrintUtils.print_extra(f"Initialized LeakyBucketMitigation with rate={self.rate}, burst_size={self.burst_size}")

    def _reshape_timing(self, original_times: list) -> list:
        """Applies leaky bucket logic to a single time sequence."""
        if not isinstance(original_times, list) or not original_times:
            return original_times

        # Ensure times are numeric
        numeric_times = [t if isinstance(t, (int, float)) else 0.0 for t in original_times]

        tokens = self.burst_size
        last_event_abs_time = 0.0
        current_abs_time = 0.0
        new_time_diffs = []
        last_sent_abs_time = 0.0 # Track absolute time of the last *sent* packet

        for t_orig in numeric_times:
            current_abs_time += max(0, t_orig) # Ensure time doesn't go backwards

            # Regenerate tokens based on time elapsed since the *last event*
            time_elapsed = current_abs_time - last_event_abs_time
            tokens_generated = time_elapsed * self.rate
            tokens = min(self.burst_size, tokens + tokens_generated)
            last_event_abs_time = current_abs_time # Update time *before* checking tokens

            if tokens >= 1.0 - 1e-9: # Use tolerance for float comparison
                # Enough tokens, send packet "immediately" relative to last sent packet
                tokens -= 1.0
                time_diff_to_append = current_abs_time - last_sent_abs_time
                new_time_diffs.append(max(0, time_diff_to_append)) # Ensure non-negative diff
                last_sent_abs_time = current_abs_time # Update last sent time
            else:
                # Not enough tokens, calculate wait time
                wait_time_needed = (1.0 - tokens) / self.rate
                tokens = 0.0  # Use up the fraction of token we had

                # Advance time to when the token becomes available
                delayed_send_time = last_event_abs_time + wait_time_needed
                last_event_abs_time = delayed_send_time # Next event starts from here
                # Current absolute time represents the time the packet *would* have arrived
                # if there was no delay. The actual send time is delayed_send_time.

                # Calculate the time difference for the delayed packet
                time_diff_to_append = delayed_send_time - last_sent_abs_time
                new_time_diffs.append(max(0, time_diff_to_append)) # Ensure non-negative diff
                last_sent_abs_time = delayed_send_time # Update last sent time

        return new_time_diffs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies leaky bucket timing reshaping to the 'time_diffs' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with 'time_diffs' potentially delayed.
        """
        PrintUtils.start_stage(f"Applying LeakyBucketMitigation (rate={self.rate}, burst={self.burst_size})")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_warning("Column 'time_diffs' not found. Skipping LeakyBucketMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        # Apply reshaping
        df_copy['time_diffs'] = df_copy['time_diffs'].apply(self._reshape_timing)
        PrintUtils.end_stage()
        return df_copy


class RandomTimeDelayMitigation(BaseMitigation):
    """
    Mitigation that adds a small, random delay to each non-zero inter-packet time.

    This adds minimal overhead but introduces jitter.
    """
    def __init__(self, max_delay: float = 0.005):
        """
        Initializes the random time delay mitigation.

        Args:
            max_delay: The maximum random delay to add (in seconds).
                       The added delay will be uniform between 0 and max_delay.
                       Defaults to 0.005 (5ms).
        """
        if max_delay < 0:
            raise ValueError("Maximum delay cannot be negative.")
        self.max_delay = max_delay
        PrintUtils.print_extra(f"Initialized RandomTimeDelayMitigation with max_delay={self.max_delay*1000:.1f}ms")

    def _add_random_delay(self, times: list) -> list:
        """Adds a small random delay to non-zero times."""
        if not isinstance(times, list): return times
        delayed_times = []
        for t in times:
            t_num = t if isinstance(t, (int, float)) else 0.0
            if t_num > 1e-9: # Use tolerance
                delay = random.uniform(0, self.max_delay)
                delayed_times.append(t_num + delay)
            else:
                delayed_times.append(t_num) # Keep zero or negative times as is
        return delayed_times

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies random delays to the 'time_diffs' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with random delays added to 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying RandomTimeDelayMitigation (max_delay={self.max_delay*1000:.1f}ms)")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_warning("Column 'time_diffs' not found. Skipping RandomTimeDelayMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        # Apply delay
        df_copy['time_diffs'] = df_copy['time_diffs'].apply(self._add_random_delay)
        PrintUtils.end_stage()
        return df_copy



class RemoveZeroTimeEntries(BaseMitigation):
    """
    Mitigation removes entries with zero time differences from the DataFrame. The
    corresponding data lengths are added to the previous entry's data length.
    """
    def __init__(self):
        """
        Initializes the random time delay mitigation.
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies random delays to the 'time_diffs' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with random delays added to 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying RemoveZeroTimeEntries")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_warning("Column 'time_diffs' not found. Skipping RemoveZeroTimeEntries.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        for index, row in df_copy.iterrows():
            times = row['time_diffs']
            sizes = row['data_lengths'] if 'data_lengths' in df_copy.columns else None

            # Remove zero time entries and adjust sizes accordingly
            new_times = []
            new_sizes = [] if sizes is not None else None
            for i, t in enumerate(times):
                if t > 1e-9:
                    new_times.append(t)
                    if new_sizes is not None:
                        new_sizes.append(sizes[i])
                elif new_sizes is not None and new_sizes:
                    # Add size to the last valid entry
                    new_sizes[-1] += sizes[i] if sizes is not None else 0

        # Update the DataFrame with the new lists
        df_copy.at[index, 'time_diffs'] = new_times
        if new_sizes is not None:
            df_copy.at[index, 'data_lengths'] = new_sizes
        
        PrintUtils.end_stage()
        return df_copy

class TimeQuantizationMitigation(BaseMitigation):
    """
    Mitigation that quantizes inter-packet times by delaying packets until
    the next time bin boundary.

    This rounds *up* the effective send time, potentially adding small delays,
    but simplifying the timing pattern.
    """
    def __init__(self, bin_interval: float = 0.010):
        """
        Initializes the time quantization mitigation.

        Args:
            bin_interval: The time interval for quantization (in seconds).
                          Packet send times are rounded up to the nearest
                          multiple of this interval. Defaults to 0.010 (10ms).
        """
        if bin_interval <= 0:
            raise ValueError("Bin interval must be positive.")
        self.bin_interval = bin_interval
        PrintUtils.print_extra(f"Initialized TimeQuantizationMitigation with bin_interval={self.bin_interval*1000:.1f}ms")

    def _quantize_times(self, original_times: list) -> list:
        """Applies time quantization logic to a single time sequence."""
        if not isinstance(original_times, list) or not original_times:
            return original_times

        # Ensure times are numeric
        numeric_times = [t if isinstance(t, (int, float)) else 0.0 for t in original_times]

        new_time_diffs = []
        current_abs_time = 0.0
        last_quantized_send_time = 0.0

        for t_orig in numeric_times:
            current_abs_time += max(0, t_orig) # Actual arrival time

            # Calculate the earliest possible send time based on quantization
            quantized_send_time = math.ceil(current_abs_time / self.bin_interval) * self.bin_interval

            # Ensure send time doesn't go backwards relative to last *sent* time
            quantized_send_time = max(quantized_send_time, last_quantized_send_time)

            # Calculate the new time difference based on quantized send times
            time_diff = quantized_send_time - last_quantized_send_time
            new_time_diffs.append(max(0, time_diff)) # Append non-negative diff

            last_quantized_send_time = quantized_send_time # Update for next iteration

        return new_time_diffs

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies time quantization to the 'time_diffs' column.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with quantized 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying TimeQuantizationMitigation (interval={self.bin_interval*1000:.1f}ms)")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_warning("Column 'time_diffs' not found. Skipping TimeQuantizationMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure time_diffs are lists
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])

        # Apply quantization
        df_copy['time_diffs'] = df_copy['time_diffs'].apply(self._quantize_times)
        PrintUtils.end_stage()
        return df_copy


class MicroBurstingMitigation(BaseMitigation):
    """
    Mitigation that coalesces packets arriving within a small time window.

    If multiple packets arrive very close together, they are buffered and sent
    as a single larger packet, altering both the timing and size sequences.
    """
    def __init__(self, time_window: float = 0.005, max_packets_in_burst: int = 5):
        """
        Initializes the micro-bursting mitigation.

        Args:
            time_window: The maximum time difference (seconds) between consecutive
                         packets to be considered part of the same burst.
                         Defaults to 0.005 (5ms).
            max_packets_in_burst: The maximum number of original packets that can
                                  be coalesced into a single burst packet.
                                  Defaults to 5.
        """
        if time_window < 0:
            raise ValueError("Time window cannot be negative.")
        if max_packets_in_burst < 2:
            raise ValueError("Max packets in burst must be at least 2.")
        self.time_window = time_window
        self.max_packets_in_burst = max_packets_in_burst
        PrintUtils.print_extra(
            f"Initialized MicroBurstingMitigation with window={self.time_window*1000:.1f}ms, "
            f"max_packets={self.max_packets_in_burst}"
        )

    def _apply_bursting(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """Applies micro-bursting logic to time and size sequences."""
        if not isinstance(original_times, list) or not isinstance(original_sizes, list) or len(original_times) != len(original_sizes):
            PrintUtils.print_warning("MicroBursting requires valid, equal-length time and size lists.")
            return original_times, original_sizes # Return original if invalid input

        if not original_times:
            return [], []

        # Ensure numeric types
        times = [t if isinstance(t, (int, float)) else 0.0 for t in original_times]
        sizes = [s if isinstance(s, (int, float)) else 0 for s in original_sizes]

        new_times = []
        new_sizes = []

        i = 0
        while i < len(times):
            burst_time_sum = times[i]
            burst_size_sum = sizes[i]
            packets_in_burst = 1
            j = i + 1

            # Check subsequent packets for bursting
            while (j < len(times) and
                   times[j] <= self.time_window and
                   packets_in_burst < self.max_packets_in_burst):
                # Add packet to burst
                burst_time_sum += times[j]
                burst_size_sum += sizes[j]
                packets_in_burst += 1
                j += 1

            # Add the (potentially coalesced) packet info
            new_times.append(max(0, burst_time_sum)) # Ensure non-negative time
            new_sizes.append(int(burst_size_sum)) # Size is sum, ensure int

            # Move main index past the processed burst
            i = j

        return new_times, new_sizes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies micro-bursting to 'time_diffs' and 'data_lengths' columns.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with potentially coalesced packets.
        """
        PrintUtils.start_stage(
            f"Applying MicroBurstingMitigation (window={self.time_window*1000:.1f}ms, max_pkts={self.max_packets_in_burst})"
        )
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_warning("Columns 'time_diffs' and 'data_lengths' required. Skipping MicroBurstingMitigation.")
            PrintUtils.end_stage()
            return df_copy

        # Ensure list formats
        df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
        df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])

        # Apply bursting - needs careful application row-wise
        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            times = row['time_diffs']
            sizes = row['data_lengths']
            new_t, new_s = self._apply_bursting(times, sizes)
            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

        PrintUtils.end_stage()
        return df_copy


# --- Mitigation Factory ---

# Dictionary to map mitigation type strings to classes
_MITIGATION_CLASSES = {
    "DataPaddingMitigation": DataPaddingMitigation,
    "TimeNoiseMitigation": TimeNoiseMitigation,
    "LeakyBucketMitigation": LeakyBucketMitigation,
    "RandomTimeDelayMitigation": RandomTimeDelayMitigation,
    "TimeQuantizationMitigation": TimeQuantizationMitigation,
    "MicroBurstingMitigation": MicroBurstingMitigation,
    "RemoveZeroTimeEntries": RemoveZeroTimeEntries,
    # Add other mitigation classes here as they are created
}

def create_mitigation(mitigation_config: dict) -> BaseMitigation:
    """
    Factory function to create mitigation instances from configuration.

    Args:
        mitigation_config: A dictionary containing 'type' (the class name)
                           and 'params' (a dictionary of parameters for __init__).

    Returns:
        An instance of the specified mitigation class.

    Raises:
        ValueError: If the mitigation type is unknown or config is invalid.
    """
    mitigation_type = mitigation_config.get("type")
    params = mitigation_config.get("params", {})

    if not mitigation_type:
        raise ValueError("Mitigation configuration must include a 'type'.")

    if mitigation_type not in _MITIGATION_CLASSES:
        raise ValueError(f"Unknown mitigation type: {mitigation_type}. Available: {list(_MITIGATION_CLASSES.keys())}")

    MitigationClass = _MITIGATION_CLASSES[mitigation_type]

    try:
        # Instantiate with parameters
        mitigation_instance = MitigationClass(**params)
        return mitigation_instance
    except TypeError as e:
         raise ValueError(f"Error initializing mitigation {mitigation_type} with params {params}. Check parameter names and types. Error: {e}")
    except Exception as e:
        # Catch other potential errors during initialization
        raise ValueError(f"Unexpected error initializing mitigation {mitigation_type}: {e}")


def apply_mitigation_set(df: pd.DataFrame, mitigation_configs: list) -> pd.DataFrame:
    """
    Applies a sequence of mitigations to a DataFrame.

    Args:
        df: The input DataFrame.
        mitigation_configs: A list of mitigation configuration dictionaries.
                            Mitigations are applied in the order they appear
                            in this list.

    Returns:
        The DataFrame after all mitigations have been applied.
    """
    df_mitigated = df # Start with the original DataFrame
    if not mitigation_configs:
        PrintUtils.print_extra("No mitigations specified for this set.")
        return df_mitigated

    PrintUtils.start_stage(f"Applying mitigation set ({len(mitigation_configs)} mitigation(s))")
    for i, config in enumerate(mitigation_configs):
        try:
            mitigation = create_mitigation(config)
            PrintUtils.print_extra(f"Applying step {i+1}/{len(mitigation_configs)}: {mitigation.__class__.__name__}")
            df_mitigated = mitigation.transform(df_mitigated)
            # Optional: Add checks here if needed (e.g., check if columns still exist)
            if df_mitigated is None or not isinstance(df_mitigated, pd.DataFrame):
                 raise RuntimeError(f"Mitigation {mitigation.__class__.__name__} returned invalid result.")
        except (ValueError, RuntimeError) as e:
            PrintUtils.print_error(f"Error applying mitigation step {i+1} ({config.get('type', 'Unknown')}): {e}")
            PrintUtils.end_stage(fail_message="Mitigation set application failed.")
            # Depending on desired behavior, you might return original df or raise further
            return df # Return the DataFrame as it was before the failing step
        except Exception as e:
             PrintUtils.print_error(f"Unexpected error during mitigation step {i+1} ({config.get('type', 'Unknown')}): {e}")
             PrintUtils.end_stage(fail_message="Mitigation set application failed due to unexpected error.")
             import traceback
             traceback.print_exc()
             return df # Return the DataFrame as it was before the failing step


    PrintUtils.end_stage() # End stage for the whole set
    return df_mitigated

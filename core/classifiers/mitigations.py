# core/mitigations.py

import abc
from collections import deque
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
    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies the mitigation logic to the DataFrame.

        Args:
            df: The input DataFrame with 'time_diffs' and 'data_lengths' columns.
                These columns should contain lists of numbers.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

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
            PrintUtils.print_extra(f"Could not parse column '{series.name}' elements as lists: {e}. Mitigation might fail or be skipped.")
            # Return original but ensure elements are lists (even if empty on failure)
            return series.apply(lambda x: x if isinstance(x, list) else [])


class PacketInjectionMitigation(BaseMitigation):
    """
    Mitigation that probabilistically injects noise packets into the sequence,
    preserving the total time duration of the original sequence.

    Injection opportunities occur based on a multiple of the median non-zero
    inter-packet time calculated across the dataset. When an injection occurs,
    it splits the time interval since the last emitted packet. The time difference
    of the injected packet reflects the time elapsed up to the injection point,
    and subsequent time differences are adjusted accordingly.
    The size of the injected packet is determined by a configurable method.
    """
    def __init__(self,
                 injection_probability: float = 0.1,
                 median_time_multiplier: float = 1.0,
                 size_method: str = 'median',
                 fixed_size: int = 64,
                 size_range: tuple[int, int] = (32, 128)):
        """
        Initializes the packet injection mitigation.

        Args:
            injection_probability: The probability (0.0 to 1.0) of injecting a
                noise packet when an injection opportunity arises. Defaults to 0.1.
            median_time_multiplier: Determines the time interval between injection
                opportunities, calculated as multiplier * median_nonzero_time.
                Defaults to 1.0.
            size_method: Method to determine the size of injected packets.
                Options: 'median', 'fixed', 'range'. Defaults to 'median'.
                - 'median': Use the median non-zero packet size from the dataset.
                - 'fixed': Use the fixed_size parameter.
                - 'range': Use a random integer between size_range[0] and size_range[1].
            fixed_size: The fixed size to use if size_method is 'fixed'. Defaults to 64.
            size_range: The tuple (min_size, max_size) for random size selection
                        if size_method is 'range'. Defaults to (32, 128).
        """
        if not 0.0 <= injection_probability <= 1.0:
            raise ValueError("Injection probability must be between 0.0 and 1.0.")
        if median_time_multiplier <= 0:
            raise ValueError("Median time multiplier must be positive.")
        if size_method not in ['median', 'fixed', 'range']:
            raise ValueError("Invalid size_method. Choose 'median', 'fixed', or 'range'.")
        if size_method == 'fixed' and fixed_size < 0:
            raise ValueError("Fixed size cannot be negative.")
        if size_method == 'range':
            if not (isinstance(size_range, tuple) and len(size_range) == 2 and
                    0 <= size_range[0] <= size_range[1]):
                raise ValueError("Size range must be a tuple of two non-negative integers (min, max) with min <= max.")

        self.injection_probability = injection_probability
        self.median_time_multiplier = median_time_multiplier
        self.size_method = size_method
        self.fixed_size = fixed_size
        self.size_range = size_range

        # These will be calculated in transform() based on the dataset
        self.median_nonzero_time = None
        self.median_packet_size = None

        PrintUtils.print_extra(
            f"Initialized PacketInjectionMitigation with prob={self.injection_probability:.2f}, "
            f"time_mult={self.median_time_multiplier:.2f}, size_method='{self.size_method}'"
        )

    def _calculate_median_nonzero_time(self, series: pd.Series) -> float:
        """Calculates the median of all non-zero time values across all lists."""
        try:
            # Flatten lists, filter numeric, keep non-zeros
            all_times = [
                float(time) for sublist in series if isinstance(sublist, list)
                for time in sublist if isinstance(time, (int, float)) and not math.isnan(time)
            ]
            non_zero_times = [time for time in all_times if time > 0.005]

            if not non_zero_times:
                return 0.0
            return float(np.median(non_zero_times))
        except Exception as e:
            PrintUtils.print_extra(f"Could not calculate median non-zero time: {e}")
            return 0.0

    def _calculate_median_size(self, series: pd.Series) -> int:
        """Calculates the median of all non-zero packet sizes across all lists."""
        try:
            # Flatten lists, filter numeric, keep non-zeros
            all_sizes = [
                int(size) for sublist in series if isinstance(sublist, list)
                for size in sublist if isinstance(size, (int, float)) and not math.isnan(size)
            ]
            non_zero_sizes = [size for size in all_sizes if size > 0]

            if not non_zero_sizes:
                return 0 # Default to 0 if no positive sizes found
            # Use np.ceil to ensure integer, leaning towards slightly larger median if needed
            return int(np.ceil(np.median(non_zero_sizes)))
        except Exception as e:
            PrintUtils.print_extra(f"Could not calculate median packet size: {e}")
            return 0 # Default to 0 on error

    def _get_injection_size(self) -> int:
        """Determines the size of the packet to be injected based on configuration."""
        if self.size_method == 'median':
            # Use the pre-calculated median size (ensure it's non-negative)
            return max(0, self.median_packet_size if self.median_packet_size is not None else 0)
        elif self.size_method == 'fixed':
            return self.fixed_size
        elif self.size_method == 'range':
            return random.randint(self.size_range[0], self.size_range[1])
        else:
            # Fallback, though validation should prevent this
            return 0

    def _apply_injection(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """
        Applies probabilistic packet injection logic to time and size sequences,
        preserving the total original time duration.

        Args:
            original_times: List of original inter-packet time differences.
            original_sizes: List of original data lengths.

        Returns:
            A tuple containing the new list of time differences and the new list
            of data lengths after applying the mitigation.
        """
        # --- Input Validation and Pre-checks ---
        if not isinstance(original_times, list) or not isinstance(original_sizes, list):
            PrintUtils.print_extra("PacketInjectionMitigation received non-list input.")
            return original_times, original_sizes
        if len(original_times) != len(original_sizes):
            PrintUtils.print_extra("PacketInjectionMitigation requires time/size lists of equal length.")
            return original_times, original_sizes
        if not original_times:
            return [], []

        # Check if necessary median values were calculated
        if self.median_nonzero_time is None or self.median_nonzero_time <= 0.005:
            PrintUtils.print_extra("Median non-zero time is zero or invalid. Skipping injection.")
            return original_times, original_sizes
        if self.size_method == 'median' and (self.median_packet_size is None or self.median_packet_size < 0):
             PrintUtils.print_extra("Median packet size is invalid for 'median' size method. Skipping injection.")
             return original_times, original_sizes

        # Ensure numeric types
        times_numeric = [(float(t) if isinstance(t, (int, float)) and not math.isnan(t) else 0.0) for t in original_times]
        sizes_numeric = [(int(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0) for s in original_sizes]

        # --- Revised Injection Logic using Time Buffer ---
        new_times = []
        new_sizes = []
        time_since_last_injection_check = 0.0
        time_buffer = 0.0 # Accumulates time since the last *emitted* packet
        # The time threshold for triggering an injection check
        injection_check_threshold = self.median_nonzero_time * self.median_time_multiplier
        if injection_check_threshold <= 0.001: # Avoid issues with zero or tiny thresholds
             PrintUtils.print_extra("Injection check threshold is too small. Skipping injection.")
             return original_times, original_sizes

        n = len(times_numeric)
        for i in range(n):
            t_orig = times_numeric[i]       # Original time difference leading to this packet
            size_orig = sizes_numeric[i]    # Original size of this packet
            current_interval_remaining = t_orig # Time within this original interval to process

            # Process the current original time interval, checking for injection points
            while current_interval_remaining > 0.005:
                # Time until the next scheduled injection check is due
                time_to_next_check = max(0.0, injection_check_threshold - time_since_last_injection_check)

                # Determine if the next check point falls within the remaining part of this interval
                if time_to_next_check <= current_interval_remaining + 0.005:
                    # --- Injection Check Point Reached ---
                    time_chunk_consumed = time_to_next_check # Process time up to the check point
                    time_buffer += time_chunk_consumed # Add this time chunk to the buffer
                    time_since_last_injection_check = 0.0 # Reset the check timer

                    # Probabilistically decide whether to inject
                    if random.random() < self.injection_probability:
                        # --- Inject Packet ---
                        inj_size = self._get_injection_size()
                        # Emit the buffered time as the time diff for the injected packet
                        new_times.append(max(0.0, time_buffer))
                        new_sizes.append(inj_size)
                        time_buffer = 0.0 # Reset buffer after emission
                    # Else (No Inject): Do nothing, time remains in the buffer.

                else:
                    # --- End of Interval Reached Before Next Check Point ---
                    time_chunk_consumed = current_interval_remaining # Process the rest of the interval
                    time_buffer += time_chunk_consumed # Add this remaining time to the buffer
                    time_since_last_injection_check += time_chunk_consumed

                # Reduce the remaining time in the current original interval
                current_interval_remaining -= time_chunk_consumed

            # --- Emit the Original Packet ---
            # After processing the entire t_orig interval, the time remaining in the buffer
            # belongs to the original packet.
            # Add the original packet using the accumulated buffer time.
            # Only add if the buffer has time or if the original size was non-zero
            # (e.g., to preserve zero-time ACK packets).
            if time_buffer > 0.005 or size_orig > 0:
                new_times.append(max(0.0, time_buffer))
                new_sizes.append(size_orig)
                time_buffer = 0.0 # Reset buffer after emission

        # Final check: Ensure total time is conserved (within float precision)
        original_total_time = sum(t for t in times_numeric if t > 0.005)
        new_total_time = sum(t for t in new_times if t > 0.005)
        if not math.isclose(original_total_time, new_total_time, abs_tol=0.1):
             PrintUtils.print_extra(
                 f"Potential time conservation issue: Original sum={original_total_time:.6f}, "
                 f"New sum={new_total_time:.6f}. Difference={original_total_time - new_total_time:.6f}"
            )
             # This warning helps catch future regressions or edge cases.

        return new_times, new_sizes


    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies probabilistic packet injection to the 'time_diffs' and
        'data_lengths' columns of the DataFrame.

        Args:
            df: The input DataFrame. Must contain 'time_diffs' and 'data_lengths'
                columns with list-like entries.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            A new DataFrame with the mitigation applied.
        """
        mitigation_name = self.__class__.__name__
        PrintUtils.start_stage(f"Applying {mitigation_name}")
        df_copy = df.copy()

        # --- Column Checks ---
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_extra(f"Columns 'time_diffs' and 'data_lengths' required. Skipping {mitigation_name}.")
            PrintUtils.end_stage()
            return df_copy

        # --- Ensure List Format ---
        try:
            df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
            df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])
        except Exception as e:
            PrintUtils.print_error(f"Error ensuring list format during {mitigation_name}: {e}. Skipping mitigation.")
            PrintUtils.end_stage(fail_message="List format conversion failed.")
            return df # Return original df on format error

        # --- Calculate Dataset-wide Medians ---
        # Use pre-calculated median time delta if available, otherwise calculate it
        if median_time_delta is not None and median_time_delta > 0.005:
            self.median_nonzero_time = median_time_delta
            PrintUtils.print_extra(f"Using pre-calculated median time delta: {self.median_nonzero_time:.6f} s")
        else:
            self.median_nonzero_time = self._calculate_median_nonzero_time(df_copy['time_diffs'])
            PrintUtils.print_extra(f"Calculated median non-zero time: {self.median_nonzero_time:.6f} s")

        if self.size_method == 'median':
            # Use pre-calculated median data size if available, otherwise calculate it
            if median_data_size is not None and median_data_size >= 0:
                self.median_packet_size = int(np.ceil(median_data_size))
                PrintUtils.print_extra(f"Using pre-calculated median data size: {self.median_packet_size} bytes")
            else:
                self.median_packet_size = self._calculate_median_size(df_copy['data_lengths'])
                PrintUtils.print_extra(f"Calculated median non-zero packet size: {self.median_packet_size} bytes")

        # --- Apply Mitigation Row-wise ---
        # Check if median calculations are valid before proceeding
        can_apply_mitigation = self.median_nonzero_time is not None and self.median_nonzero_time > 0.005
        if self.size_method == 'median':
             can_apply_mitigation &= (self.median_packet_size is not None and self.median_packet_size >= 0)

        if not can_apply_mitigation:
             PrintUtils.print_extra(f"Cannot apply {mitigation_name} due to invalid median calculations. Returning unmodified data.")
             PrintUtils.end_stage()
             return df_copy

        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            original_times = row['time_diffs']
            original_sizes = row['data_lengths']

            try:
                new_t, new_s = self._apply_injection(original_times, original_sizes)
            except Exception as e:
                PrintUtils.print_extra(f"Error applying injection logic for row {index}: {e}. Skipping row modification.")
                new_t, new_s = original_times, original_sizes # Keep original on error

            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        # --- Update DataFrame Columns ---
        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

        PrintUtils.end_stage()
        return df_copy


class AdaptivePacketInjectionMitigation(BaseMitigation):
    """
    Mitigation that probabilistically injects noise packets, adapting the
    injection timing based on recently observed inter-packet intervals within
    the current sequence, while preserving the total original time duration.

    Injection opportunities are dynamically scheduled based on a running average
    of recent non-negligible inter-packet times multiplied by a factor.
    This allows the mitigation to adjust to faster or slower parts of a sequence.
    """
    def __init__(self,
                 injection_probability: float = 0.1,
                 median_time_multiplier: float = 1.0,
                 adaptive_window_size: int = 10,
                 adaptive_min_samples: int = 3,
                 time_epsilon: float = 0.002, # Threshold below which times are ignored for adaptation
                 size_method: str = 'median',
                 fixed_size: int = 64,
                 size_range: tuple[int, int] = (32, 128)):
        """
        Initializes the adaptive packet injection mitigation.

        Args:
            injection_probability: Probability (0.0-1.0) of injecting a packet
                at an opportunity. Defaults to 0.1.
            median_time_multiplier: Factor multiplied by the current adaptive
                time estimate to set the injection check interval. Defaults to 1.0.
            adaptive_window_size: The number of recent relevant time intervals
                to consider for the adaptive timing calculation. Defaults to 10.
            adaptive_min_samples: The minimum number of relevant time intervals
                needed in the window to switch from global median to adaptive
                timing. Defaults to 3.
            time_epsilon: Time differences below this value (in seconds) are
                considered negligible and ignored for adaptive calculations and
                initial median calculation. Defaults to 0.005 (5ms).
            size_method: Method for determining injected packet size ('median',
                'fixed', 'range'). Defaults to 'median'.
            fixed_size: Size used if size_method is 'fixed'. Defaults to 64.
            size_range: Tuple (min, max) used if size_method is 'range'.
                        Defaults to (32, 128).
        """
        # --- Input Validation ---
        if not 0.0 <= injection_probability <= 1.0:
            raise ValueError("Injection probability must be between 0.0 and 1.0.")
        if median_time_multiplier <= 0:
            raise ValueError("Median time multiplier must be positive.")
        if not isinstance(adaptive_window_size, int) or adaptive_window_size <= 0:
            raise ValueError("Adaptive window size must be a positive integer.")
        if not isinstance(adaptive_min_samples, int) or adaptive_min_samples <= 0:
             raise ValueError("Adaptive minimum samples must be a positive integer.")
        if adaptive_min_samples > adaptive_window_size:
            raise ValueError("Adaptive minimum samples cannot be larger than adaptive window size.")
        if time_epsilon < 0:
            raise ValueError("Time epsilon cannot be negative.")
        if size_method not in ['median', 'fixed', 'range']:
            raise ValueError("Invalid size_method. Choose 'median', 'fixed', or 'range'.")
        if size_method == 'fixed' and fixed_size < 0:
            raise ValueError("Fixed size cannot be negative.")
        if size_method == 'range':
            if not (isinstance(size_range, tuple) and len(size_range) == 2 and
                    0 <= size_range[0] <= size_range[1]):
                raise ValueError("Size range must be a tuple (min_size, max_size) with min_size <= max_size and both non-negative.")

        # --- Store Parameters ---
        self.injection_probability = injection_probability
        self.median_time_multiplier = median_time_multiplier
        self.adaptive_window_size = adaptive_window_size
        self.adaptive_min_samples = adaptive_min_samples
        self.time_epsilon = time_epsilon
        self.size_method = size_method
        self.fixed_size = fixed_size
        self.size_range = size_range

        # --- Internal State (Calculated during transform) ---
        self.global_median_nonzero_time = None
        self.median_packet_size = None # Only calculated if size_method is 'median'

        PrintUtils.print_extra(
            f"Initialized AdaptivePacketInjectionMitigation with prob={self.injection_probability:.2f}, "
            f"time_mult={self.median_time_multiplier:.2f}, window={self.adaptive_window_size}, "
            f"min_samples={self.adaptive_min_samples}, epsilon={self.time_epsilon:.4f}, "
            f"size_method='{self.size_method}'"
        )

    def _calculate_median_size(self, series: pd.Series) -> int:
        """Calculates the median of all non-zero packet sizes across all lists."""
        try:
            all_sizes = [
                int(size) for sublist in series if isinstance(sublist, list)
                for size in sublist if isinstance(size, (int, float)) and not math.isnan(size)
            ]
            non_zero_sizes = [size for size in all_sizes if size > 0]

            if not non_zero_sizes:
                PrintUtils.print_extra("No non-zero sizes found for median calculation.")
                return 0
            return int(np.ceil(np.median(non_zero_sizes)))
        except Exception as e:
            PrintUtils.print_extra(f"Could not calculate median packet size: {e}")
            return 0

    def _get_injection_size(self) -> int:
        """Determines the size of the packet to be injected based on configuration."""
        if self.size_method == 'median':
            return max(0, self.median_packet_size if self.median_packet_size is not None else 0)
        elif self.size_method == 'fixed':
            return self.fixed_size
        elif self.size_method == 'range':
            return random.randint(self.size_range[0], self.size_range[1])
        else: # Fallback
            return 0

    def _apply_adaptive_injection(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """
        Applies adaptive probabilistic packet injection logic to time and size sequences.

        Args:
            original_times: List of original inter-packet time differences.
            original_sizes: List of original data lengths.

        Returns:
            A tuple containing the new list of time differences and the new list
            of data lengths after applying the mitigation.
        """
        # --- Input Validation and Pre-checks ---
        if not isinstance(original_times, list) or not isinstance(original_sizes, list):
            PrintUtils.print_extra("AdaptivePacketInjectionMitigation received non-list input.")
            return original_times, original_sizes
        if len(original_times) != len(original_sizes):
            PrintUtils.print_extra("AdaptivePacketInjectionMitigation requires time/size lists of equal length.")
            return original_times, original_sizes
        if not original_times:
            return original_times, original_sizes

        # Check if global median was calculated and is valid
        if self.global_median_nonzero_time is None or self.global_median_nonzero_time <= self.time_epsilon:
            PrintUtils.print_extra("Global median non-zero time is invalid or too small. Skipping injection.")
            return original_times, original_sizes
        if self.size_method == 'median' and (self.median_packet_size is None or self.median_packet_size < 0):
             PrintUtils.print_extra("Median packet size is invalid for 'median' size method. Skipping injection.")
             return original_times, original_sizes

        # Ensure numeric types
        times_numeric = [(float(t) if isinstance(t, (int, float)) and not math.isnan(t) else 0.0) for t in original_times]
        sizes_numeric = [(int(s) if isinstance(s, (int, float)) and not math.isnan(s) else 0) for s in original_sizes]

        # --- Adaptive Injection Logic ---
        new_times = []
        new_sizes = []
        time_since_last_injection_check = 0.0
        time_buffer = 0.0 # Accumulates time since the last emitted packet
        # Moving window for recent relevant times
        recent_relevant_times = deque(maxlen=self.adaptive_window_size)
        current_adaptive_time_base = self.global_median_nonzero_time # Start with global

        n = len(times_numeric)
        for i in range(n):
            t_orig = times_numeric[i]       # Original time difference leading to this packet
            size_orig = sizes_numeric[i]    # Original size of this packet

            # --- Update Adaptive Timing ---
            # Add the *previous* interval's time to the window if relevant
            if i > 0:
                prev_time = times_numeric[i-1]
                if prev_time > self.time_epsilon:
                    recent_relevant_times.append(prev_time)

            # Calculate current adaptive time base (use global until enough samples)
            if len(recent_relevant_times) >= self.adaptive_min_samples:
                current_adaptive_time_base = float(np.mean(recent_relevant_times))
            else:
                current_adaptive_time_base = self.global_median_nonzero_time

            # Calculate the injection check threshold for this step
            injection_check_threshold = current_adaptive_time_base * self.median_time_multiplier
            # Ensure threshold is not impractically small
            if injection_check_threshold <= self.time_epsilon / 2.0:
                injection_check_threshold = self.global_median_nonzero_time * self.median_time_multiplier
                if injection_check_threshold <= self.time_epsilon / 2.0:
                    injection_check_threshold = float('inf')  # Disable injection for this interval

            # --- Process Current Interval ---
            current_interval_remaining = t_orig
            while current_interval_remaining > self.time_epsilon: # Only check within significant time intervals
                time_to_next_check = max(0.0, injection_check_threshold - time_since_last_injection_check)

                if time_to_next_check <= current_interval_remaining + self.time_epsilon: # Allow for float tolerance
                    # Injection Check Point Reached
                    time_chunk_consumed = time_to_next_check
                    time_buffer += time_chunk_consumed
                    time_since_last_injection_check = 0.0

                    if random.random() < self.injection_probability:
                        # Inject Packet
                        inj_size = self._get_injection_size()
                        new_times.append(max(0.0, time_buffer))
                        new_sizes.append(inj_size)
                        time_buffer = 0.0 # Reset buffer
                    # When no injection occurs, time_buffer keeps the accumulated time

                    current_interval_remaining -= time_chunk_consumed
                else:
                    # End of Interval Reached Before Next Check Point
                    time_chunk_consumed = current_interval_remaining
                    time_buffer += time_chunk_consumed
                    time_since_last_injection_check += time_chunk_consumed
                    current_interval_remaining = 0.0  # Exit loop

            # Add any remaining part of the interval (or the whole interval if < epsilon) to buffer
            if current_interval_remaining > 0:
                time_buffer += current_interval_remaining

            # --- Emit the Original Packet ---
            # Use the accumulated time in the buffer.
            if time_buffer > self.time_epsilon or size_orig > 0:
                new_times.append(max(0.0, time_buffer))
                new_sizes.append(size_orig)
                time_buffer = 0.0 # Reset buffer

        # --- Final Time Conservation Check (Optional but Recommended) ---
        original_total_time = sum(t for t in times_numeric if t > self.time_epsilon)
        new_total_time = sum(t for t in new_times if t > self.time_epsilon)
        if not math.isclose(original_total_time, new_total_time, abs_tol=0.01):
             PrintUtils.print_extra(
                 f"[AdaptiveInject] Potential time conservation issue: "
                 f"Original sum={original_total_time:.6f}, New sum={new_total_time:.6f}. "
                 f"Diff={original_total_time - new_total_time:.6f}"
             )

        return new_times, new_sizes

    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies adaptive probabilistic packet injection to the DataFrame.

        Args:
            df: The input DataFrame with 'time_diffs' and 'data_lengths'.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            A new DataFrame with the mitigation applied.
        """
        mitigation_name = self.__class__.__name__
        PrintUtils.start_stage(f"Applying {mitigation_name}")
        df_copy = df.copy()

        # --- Column Checks ---
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_extra(f"Columns 'time_diffs'/'data_lengths' required. Skipping {mitigation_name}.")
            PrintUtils.end_stage()
            return df_copy

        # --- Ensure List Format ---
        try:
            df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
            df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])
        except Exception as e:
            PrintUtils.print_error(f"Error ensuring list format during {mitigation_name}: {e}. Skipping mitigation.")
            PrintUtils.end_stage(fail_message="List format conversion failed.")
            return df

        # --- Calculate Global Medians (Needed for initialization and potentially size) ---
        self.global_median_nonzero_time = median_time_delta
        PrintUtils.print_extra(f"Calculated global median non-negligible time (>{self.time_epsilon:.4f}s): {self.global_median_nonzero_time:.6f} s")

        if self.size_method == 'median':
            self.median_packet_size = self._calculate_median_size(df_copy['data_lengths'])
            PrintUtils.print_extra(f"Calculated median non-zero packet size: {self.median_packet_size} bytes")

        # --- Check if Mitigation Can Run ---
        can_apply_mitigation = self.global_median_nonzero_time is not None and self.global_median_nonzero_time > self.time_epsilon
        if self.size_method == 'median':
            can_apply_mitigation &= (self.median_packet_size is not None and self.median_packet_size >= 0)

        if not can_apply_mitigation:
             PrintUtils.print_extra(f"Cannot apply {mitigation_name} due to invalid global calculations. Returning unmodified data.")
             PrintUtils.end_stage()
             return df_copy

        # --- Apply Mitigation Row-wise ---
        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            original_times = row['time_diffs']
            original_sizes = row['data_lengths']

            try:
                new_t, new_s = self._apply_adaptive_injection(original_times, original_sizes)
            except Exception as e:
                PrintUtils.print_extra(f"Error applying adaptive injection to row {index}: {e}. Using original data.")
                new_t, new_s = original_times, original_sizes

            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        # --- Update DataFrame Columns ---
        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

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

    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies random delays to the 'time_diffs' column.

        Args:
            df: The input DataFrame.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            DataFrame with random delays added to 'time_diffs'.
        """
        PrintUtils.start_stage(f"Applying RemoveZeroTimeEntries")
        df_copy = df.copy()
        if 'time_diffs' not in df_copy.columns:
            PrintUtils.print_extra("Column 'time_diffs' not found. Skipping RemoveZeroTimeEntries.")
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
                if t > 0.005:
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


class PacketBatchingMitigation(BaseMitigation):
    """
    Mitigation that batches consecutive packets together.

    Combines N consecutive packets into a single packet by summing their
    inter-packet time differences and data lengths. This reduces the sequence
    length and alters the timing and size patterns.
    """
    def __init__(self, batch_size: int = 2):
        """
        Initializes the packet batching mitigation.

        Args:
            batch_size: The number of consecutive packets to combine into one.
                        Must be an integer >= 2. Defaults to 2.
        """
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("Batch size must be an integer greater than or equal to 2.")
        self.batch_size = batch_size
        PrintUtils.print_extra(f"Initialized PacketBatchingMitigation with batch_size={self.batch_size}")

    def _apply_batching(self, original_times: list, original_sizes: list) -> tuple[list, list]:
        """
        Applies batching logic to a single pair of time and size sequences.

        Args:
            original_times: List of original inter-packet time differences.
            original_sizes: List of original data lengths.

        Returns:
            A tuple containing the new list of batched time differences and the
            new list of batched data lengths.
        """
        # --- Input Validation ---
        if not isinstance(original_times, list) or not isinstance(original_sizes, list):
            PrintUtils.print_extra("PacketBatchingMitigation received non-list input.")
            return original_times, original_sizes
        if len(original_times) != len(original_sizes):
            PrintUtils.print_extra("PacketBatchingMitigation requires time/size lists of equal length.")
            return original_times, original_sizes
        if not original_times:
            return [], []

        n = len(original_times)
        new_times = []
        new_sizes = []

        # --- Batching Loop ---
        for i in range(0, n, self.batch_size):
            # Define the slice for the current batch
            start_index = i
            end_index = min(i + self.batch_size, n) # Handle potential partial batch at the end

            # Extract the batch data
            time_batch = original_times[start_index:end_index]
            size_batch = original_sizes[start_index:end_index]

            # Sum the time differences in the batch
            # Ensure numeric conversion and handle potential NaNs
            batched_time = sum(
                float(t) for t in time_batch
                if isinstance(t, (int, float)) and not math.isnan(t)
            )
            # Sum the data lengths in the batch
            batched_size = sum(
                int(s) for s in size_batch
                if isinstance(s, (int, float)) and not math.isnan(s)
            )

            # Append the batched results
            new_times.append(max(0.0, batched_time)) # Ensure non-negative time
            new_sizes.append(int(batched_size))

        return new_times, new_sizes

    def transform(self, df: pd.DataFrame, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
        """
        Applies packet batching to the 'time_diffs' and 'data_lengths' columns.

        Args:
            df: The input DataFrame. Must contain 'time_diffs' and 'data_lengths'
                columns with list-like entries.
            median_time_delta: Pre-calculated median time delta from the full dataset.
            median_data_size: Pre-calculated median data size from the full dataset.

        Returns:
            A new DataFrame with the packet batching mitigation applied.
        """
        mitigation_name = self.__class__.__name__
        PrintUtils.start_stage(f"Applying {mitigation_name} (batch_size={self.batch_size})")
        df_copy = df.copy()

        # --- Column Checks ---
        if 'time_diffs' not in df_copy.columns or 'data_lengths' not in df_copy.columns:
            PrintUtils.print_extra(f"Columns 'time_diffs' and 'data_lengths' required. Skipping {mitigation_name}.")
            PrintUtils.end_stage()
            return df_copy

        # --- Ensure List Format ---
        try:
            df_copy['time_diffs'] = self._ensure_list_format(df_copy['time_diffs'])
            df_copy['data_lengths'] = self._ensure_list_format(df_copy['data_lengths'])
        except Exception as e:
            PrintUtils.print_error(f"Error ensuring list format during {mitigation_name}: {e}. Skipping mitigation.")
            PrintUtils.end_stage(fail_message="List format conversion failed.")
            return df # Return original df on format error

        # --- Apply Mitigation Row-wise ---
        new_times_list = []
        new_sizes_list = []
        for index, row in df_copy.iterrows():
            original_times = row['time_diffs']
            original_sizes = row['data_lengths']

            try:
                new_t, new_s = self._apply_batching(original_times, original_sizes)
            except Exception as e:
                PrintUtils.print_extra(f"Error applying batching logic for row {index}: {e}. Skipping row modification.")
                new_t, new_s = original_times, original_sizes # Keep original on error

            new_times_list.append(new_t)
            new_sizes_list.append(new_s)

        # --- Update DataFrame Columns ---
        df_copy['time_diffs'] = new_times_list
        df_copy['data_lengths'] = new_sizes_list

        PrintUtils.end_stage()
        return df_copy

# --- Mitigation Factory ---

# Dictionary to map mitigation type strings to classes
_MITIGATION_CLASSES = {
    "RemoveZeroTimeEntries": RemoveZeroTimeEntries,
    "PacketInjectionMitigation": PacketInjectionMitigation,
    "AdaptivePacketInjectionMitigation": AdaptivePacketInjectionMitigation,
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


def apply_mitigation_set(df: pd.DataFrame, mitigation_configs: list, median_time_delta: float = None, median_data_size: float = None) -> pd.DataFrame:
    """
    Applies a sequence of mitigations to a DataFrame.

    Args:
        df: The input DataFrame.
        mitigation_configs: A list of mitigation configuration dictionaries.
                            Mitigations are applied in the order they appear
                            in this list.
        median_time_delta: Pre-calculated median time delta from the full dataset.
        median_data_size: Pre-calculated median data size from the full dataset.

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
            df_mitigated = mitigation.transform(df_mitigated, median_time_delta=median_time_delta, median_data_size=median_data_size)
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

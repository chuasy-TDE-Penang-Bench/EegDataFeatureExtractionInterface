import numpy as np
import pandas as pd
import os
from scipy.fft import fft
from scipy.signal import cheby2, filtfilt

def process(self):
    # Parameters
    window_size = 128
    overlap = 50  # 50% overlap
    window_width = int(np.ceil(window_size - window_size * overlap / 100))  # Step size
    Fs = 256  # Sampling frequency
    wn = Fs / 2  # Nyquist frequency

    # Define frequency bands and filters
    frequency_bands = ['delta', 'theta', 'alpha', 'beta']
    filters = {
        'delta': cheby2(2, 40, [0.5 / wn, 4 / wn], btype='band'),
        'theta': cheby2(2, 40, [4 / wn, 7 / wn], btype='band'),
        'alpha': cheby2(2, 40, [8 / wn, 12 / wn], btype='band'),
        'beta': cheby2(2, 40, [13 / wn, 30 / wn], btype='band')
    }

    # Hemisphere mapping (1-based indices)
    hemisphere_map = {
        'left': np.array(
            [1, 3, 4, 8, 9, 12, 13, 17, 18, 19, 23, 24, 28, 29, 33, 35, 36, 39, 42, 45, 46, 47, 48, 49, 55, 57, 58,
             61]) - 1,
        'right': np.array(
            [2, 6, 7, 10, 11, 15, 16, 20, 21, 22, 26, 27, 31, 32, 34, 37, 38, 41, 43, 50, 51, 52, 53, 54, 56, 59,
             60, 62]) - 1
    }

    # Initialize GUI elements
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    self.progress_bar['maximum'] = len(self.file_list)
    self.progress_bar['value'] = 0

    # Process each EEG datasets
    for idx, file in enumerate(self.file_list):
        self.progress_bar['value'] = idx + 1
        self.root.update_idletasks()
        self.log(f"Processing: {os.path.normpath(file)}")

        # Read EEG data
        df = pd.read_csv(file, header=None).values
        num_channels, num_samples = df.shape

        # Create output directory
        file_base_name = os.path.splitext(os.path.basename(file))[0]
        file_output_dir = os.path.join(output_dir, file_base_name)
        os.makedirs(file_output_dir, exist_ok=True)

        # Storage for PSD results per frequency band
        psd_results = {band: [] for band in frequency_bands}

        # Process each EEG channel
        for ch_index in range(num_channels):
            channel_data = df[ch_index, :]
            total_frames = int(np.ceil(num_samples / window_width)) - 1

            # Storage for channel-specific PSD results per frequency band
            psd_channel = {band: [] for band in frequency_bands}

            # Process each data frame
            for frame in range(total_frames + 1):
                start, end = frame * window_width, frame * window_width + window_size
                if end > num_samples:
                    break

                data_frame = channel_data[start:end]

                # Apply filter and computer fft
                for band, (b, a) in filters.items():
                    filtered_data = filtfilt(b, a, data_frame)
                    fft_result = np.abs(fft(filtered_data, Fs)[:50])
                    psd_channel[band].append(np.max(np.square(fft_result)))

            # Store PSD results per band
            for band in frequency_bands:
                psd_results[band].append(psd_channel[band])

        # Save PSD results per band to csv
        for band in frequency_bands:
            band_file_path = os.path.join(file_output_dir, f'{band}_psd.csv')
            pd.DataFrame(np.array(psd_results[band]).T).to_csv(band_file_path, index=False, header=False)
            self.log(f"Saved {band}_psd.csv to {file_output_dir}")

        # Compute hemisphere-based summary
        summary_data = {band: {'left': [], 'right': []} for band in frequency_bands}
        for band in frequency_bands:
            band_matrix = np.array(psd_results[band])  # Shape: (62, frames)
            left_values = band_matrix[hemisphere_map['left'], :]
            right_values = band_matrix[hemisphere_map['right'], :]

            for frame in range(band_matrix.shape[1]):
                summary_data[band]['left'].append(np.mean(left_values[:, frame]))
                summary_data[band]['right'].append(np.mean(right_values[:, frame]))

        # Create final summary data
        final_summary = np.hstack([
            np.column_stack([summary_data[band]['left'], summary_data[band]['right']])
            for band in frequency_bands
        ])

        # Determine emotion class from filename
        class_mapping = {'s': 'sad', 'h': 'happy', 'f': 'fear', 'n': 'neutral'}
        class_label = next((class_mapping[c] for c in class_mapping if c in file_base_name), None)
        class_column = [class_label] * final_summary.shape[0]

        # Save the final summary CSV
        columns = ["ALPHA L", "ALPHA R", "BETA L", "BETA R", "DELTA L", "DELTA R", "THETA L", "THETA R", "CLASS"]
        summary_file = os.path.join(file_output_dir, 'psd_summary.csv')
        summary_df = pd.DataFrame(np.column_stack([final_summary, class_column]), columns=columns)
        summary_df.to_csv(summary_file, index=False)
        self.log(f"Saved psd_summary.csv to {file_output_dir}")

    messagebox.showinfo("Processing Complete", "All files processed and saved successfully!")
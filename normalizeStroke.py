import os
import re
import joblib
import pandas as pd
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def process(self):
    # Parameters
    columns = ["ALPHA L", "ALPHA R", "BETA L", "BETA R", "DELTA L", "DELTA R", "THETA L", "THETA R"]

    # Initialize GUI elements
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    self.progress_bar['maximum'] = len(self.file_list)
    self.progress_bar['value'] = 0

    category = ""
    eyes_status = ""
    stacked_df = pd.DataFrame()

    # Process each EEG datasets
    for idx, file in enumerate(self.file_list):
        self.progress_bar['value'] = idx + 1
        self.root.update_idletasks()
        self.log(f"Processing: {os.path.normpath(file)}")

        file_dir = os.path.dirname(file).lower()

        # Determine eye condition (open eyes, close eyes)
        if "open eyes" in file_dir:
            eyes_status = "open eyes"
        elif "close eyes" in file_dir:
            eyes_status = "close eyes"
        else:
            self.log(f"Warning: No 'open eyes' or 'close eyes' found in {file}. Skipping.")
            continue

        # Determine category (minor, moderate, severe)
        category_match = re.search(r'(minor|moderate|severe)', file_dir)
        if not category_match:
            self.log(f"Warning: No category ('minor', 'moderate', or 'severe') found in {file}. Skipping.")
            continue
        category = category_match.group(1).lower()

        # Read EEG data and stack them
        df = pd.read_csv(file, skiprows=[0], header=None)
        stacked_df = pd.concat([stacked_df, df], ignore_index=True)





    # Load the saved scaler
    # scaler_filename = f"scaler.pkl"
    # scaler_file = os.path.join(output_dir, scaler_filename)
    # scaler = joblib.load(scaler_file)
    # self.log(f"Loaded file: {os.path.normpath(scaler_file)}")

    # # Initialize the scaler
    scaler = StandardScaler()

    # Select only numeric columns for normalization
    numeric_cols = stacked_df.select_dtypes(include=['float64', 'int64']).columns

    # Fit the scaler on training data
    scaler.fit(stacked_df[numeric_cols])

    # Save the scaler for later use
    scaler_filename = f"{category}_{eyes_status}_scaler.pkl".replace(" ", "_")
    scaler_file = os.path.join(output_dir, scaler_filename)
    joblib.dump(scaler, scaler_file)
    print(f"Saved file: {os.path.normpath(scaler_file)}")

    # Normalize only numeric columns
    df_normalized = pd.DataFrame(scaler.transform(stacked_df[numeric_cols]), columns=numeric_cols)
    # df_normalized = pd.DataFrame(scaler.transform(stacked_df), columns=stacked_df.columns)

    # Define output filename and save
    new_filename = f"{category}_{eyes_status}_test_data.csv".replace(" ", "_")
    output_file = os.path.join(output_dir, new_filename)
    df_normalized.to_csv(output_file, index=False, header=columns)
    self.log(f"Saved file: {os.path.normpath(output_file)}")

    messagebox.showinfo("Processing Complete", "All files processed and saved successfully!")
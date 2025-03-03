import os
import re
import pandas as pd
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler

def process(self):
    # Parameters
    columns = ["ALPHA L", "ALPHA R", "BETA L", "BETA R", "DELTA L", "DELTA R", "THETA L", "THETA R", "CLASS"]
    stacked_df = []

    # Initialize GUI elements
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    self.progress_bar['maximum'] = len(self.file_list)
    self.progress_bar['value'] = 0

    category = ""
    eyes_status = ""
    new_filename = ""
    stacked_df = pd.DataFrame()

    # Process each EEG datasets
    for idx, file in enumerate(self.file_list):
        self.progress_bar['value'] = idx + 1
        self.root.update_idletasks()
        self.log(f"Processing: {os.path.normpath(file)}")

        # file_dir = os.path.dirname(file).lower()
        #
        # # Determine eye condition (open eyes, close eyes)
        # if "open eyes" in file_dir:
        #     eyes_status = "open eyes"
        # elif "close eyes" in file_dir:
        #     eyes_status = "close eyes"
        # else:
        #     self.log(f"Warning: No 'open eyes' or 'close eyes' found in {file}. Skipping.")
        #     continue

        # Determine category (minor, moderate, severe)
        # category_match = re.search(r'(minor|moderate|severe)', file_dir)
        # if not category_match:
        #     self.log(f"Warning: No category ('minor', 'moderate', or 'severe') found in {file}. Skipping.")
        #     continue
        # category = category_match.group(1).lower()

        # Read EEG data and stack them
        df = pd.read_csv(file, skiprows=[0], header=None)
        stacked_df = pd.concat([stacked_df, df], ignore_index=True)

    # # Initialize the scaler
    # scaler = MinMaxScaler()
    # numeric_cols = stacked_df.select_dtypes(include=['float64', 'int64']).columns
    # # stacked_df[numeric_cols] = scaler.fit_transform(stacked_df[numeric_cols])
    #
    # # Normalize the data
    # df_normalized = pd.DataFrame(scaler.fit_transform(stacked_df), columns=stacked_df.columns)
    #
    # selected_value = self.selection_var.get()

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Select only numeric columns for normalization
    numeric_cols = stacked_df.select_dtypes(include=['float64', 'int64']).columns

    # Normalize only numeric columns
    df_normalized = pd.DataFrame(scaler.fit_transform(stacked_df[numeric_cols]), columns=numeric_cols)

    # Concatenate normalized numeric columns with non-numeric columns
    df_normalized = pd.concat([df_normalized, stacked_df.drop(columns=numeric_cols)], axis=1)

    # Reorder columns to match the original DataFrame
    df_normalized = df_normalized[stacked_df.columns]

    # Define output filename and save
    new_filename = f"train_data.csv"
    output_file = os.path.join(output_dir, new_filename)
    df_normalized.to_csv(output_file, index=False, header=columns)

    messagebox.showinfo("Processing Complete", "All files processed and saved successfully!")
import os
import joblib
import pandas as pd
from tkinter import messagebox

from PIL.ImageOps import scale
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def process(self):
    # Parameters
    columns = ["ALPHA L", "ALPHA R", "BETA L", "BETA R", "DELTA L", "DELTA R", "THETA L", "THETA R", "CLASS"]

    # Initialize GUI elements
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    self.progress_bar['maximum'] = len(self.file_list)
    self.progress_bar['value'] = 0

    stacked_df = pd.DataFrame()
    scaler_value = self.scaler_var.get()

    # Process each EEG datasets
    for idx, file in enumerate(self.file_list):
        self.progress_bar['value'] = idx + 1
        self.root.update_idletasks()
        self.log(f"Processing: {os.path.normpath(file)}")

        # Read EEG data and stack them
        df = pd.read_csv(file, skiprows=[0], header=None)
        stacked_df = pd.concat([stacked_df, df], ignore_index=True)

    # Initialize the scaler
    if scaler_value == "Standard":
        scaler = StandardScaler()
    elif scaler_value == "MinMax":
        scaler = MinMaxScaler()

    # Select only numeric columns for normalization
    numeric_cols = stacked_df.select_dtypes(include=['float64', 'int64']).columns

    # Fit the scaler on training data
    scaler.fit(stacked_df[numeric_cols])

    # Save the scaler for later use
    scaler_filename = f"train_scaler.pkl"
    scaler_file = os.path.join(output_dir, scaler_filename)
    joblib.dump(scaler, scaler_file)
    print(f"Saved file: {os.path.normpath(scaler_file)}")

    # Normalize only numeric columns
    df_normalized = pd.DataFrame(scaler.transform(stacked_df[numeric_cols]), columns=numeric_cols)

    # Concatenate normalized numeric columns with non-numeric columns
    df_normalized = pd.concat([df_normalized, stacked_df.drop(columns=numeric_cols)], axis=1)

    # Reorder columns to match the original DataFrame
    df_normalized = df_normalized[stacked_df.columns]

    # Define output filename and save
    new_filename = f"train_data.csv"
    output_file = os.path.join(output_dir, new_filename)
    df_normalized.to_csv(output_file, index=False, header=columns)
    self.log(f"Saved file: {os.path.normpath(output_file)}")


    messagebox.showinfo("Processing Complete", "All files processed and saved successfully!")
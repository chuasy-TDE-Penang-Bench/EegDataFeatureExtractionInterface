import os
import re
import pandas as pd
from tkinter import messagebox

def process(self):
    # Parameters
    columns = ["ALPHA L", "ALPHA R", "BETA L", "BETA R", "DELTA L", "DELTA R", "THETA L", "THETA R"]

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

        # Extract 'P' value from filename
        match = re.search(r'P\d+', file)
        if not match:
            self.log(f"Warning: No 'P' value found in {file}. Skipping.")
            continue
        p_value = match.group()

        # Get file directory of the EEG dataset
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

        # Read EEG data and remove second row
        df = pd.read_csv(file, skiprows=[1])

        # Select only columns of interest
        df = df[columns]

        # Remove rows containing "=" in any cell
        df = df[~df.apply(lambda row: row.astype(str).str.contains("=").any(), axis=1)]

        # Create output directory
        output_category_dir = os.path.join(output_dir, category, eyes_status)
        os.makedirs(output_category_dir, exist_ok=True)

        # Define output filename and save
        new_filename = f"{p_value}.csv"

        output_file = os.path.join(output_category_dir, new_filename)
        df.to_csv(output_file, index=False)

        self.log(f"Saved {new_filename} to {os.path.normpath(output_category_dir)}")

    messagebox.showinfo("Processing Complete", "All files processed and saved successfully!")
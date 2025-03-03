import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Function to process and normalize csv files (train or test datasets)
def process_psd_summary(input_directory, output_directory, is_train=True):
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize an empty list to store processed data
    combined_data = []

    # For training data, handle subject folders in NormalFeatureExtraction
    if is_train:
        categories = [folder for folder in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, folder))]
    else:
        # For test data, handle severity and eyes categories
        severity_folders = ['minor', 'moderate', 'severe']
        eye_status_folders = ['Close eyes', 'Open eyes']

    # Loop through each category (subject folder for train and severity/eyes for test data)
    for category in categories if is_train else severity_folders:
        if is_train:
            # For training dataset, we have subject folders like 1-1s, 1-2f, etc.
            category_folder = os.path.join(input_directory, category)
            output_file_name = "train_data.csv"  # All train data in one CSV
        else:
            # For test dataset, we process by severity and eyes
            for eye_status in eye_status_folders:
                category_folder = os.path.join(input_directory, category, eye_status)
                output_file_name = f"{category}_{eye_status}.csv"

                category_data = []

                # Walk through the category folder to find all CSV files
                print(f"Processing category: {category_folder}")  # Debugging print
                for subdir, _, files in os.walk(category_folder):
                    for file in files:
                        # Process only the relevant files
                        if file.endswith('.csv'):  # For test data
                            file_path = os.path.join(subdir, file)

                            # Read the CSV into a DataFrame
                            print(f"Reading file: {file_path}")  # Debugging print
                            df = pd.read_csv(file_path)

                            # Check if the DataFrame is empty
                            if df.empty:
                                print(f"Warning: {file_path} is empty. Skipping.")  # Debugging print
                                continue

                            # Normalize the data (exclude non-numeric columns for normalization)
                            scaler = MinMaxScaler()
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                            # Append the normalized data for this test category
                            category_data.append(df)
                            print(f"Processed: {file_path}")  # Debugging print

                # If data exists for this category and eye status
                if category_data:
                    final_category_df = pd.concat(category_data, ignore_index=True)
                    # Write the final combined dataframe to a CSV file in the specified output directory
                    output_path = os.path.join(output_directory, output_file_name)
                    final_category_df.to_csv(output_path, index=False)
                    print(f"Final combined CSV saved to: {output_path}")
                else:
                    print(f"No valid data found for {category_folder}. Skipping.")

        # For training data, just combine all subjects into one CSV
        if is_train:
            category_data = []

            for subdir, _, files in os.walk(category_folder):
                for file in files:
                    # Process only the psd_summary.csv files
                    if file == 'psd_summary.csv':
                        file_path = os.path.join(subdir, file)
                        # Read the CSV into a DataFrame
                        print(f"Reading file: {file_path}")  # Debugging print
                        df = pd.read_csv(file_path)

                        # Check if the DataFrame is empty
                        if df.empty:
                            print(f"Warning: {file_path} is empty. Skipping.")  # Debugging print
                            continue

                        # Normalize the data (exclude non-numeric columns for normalization)
                        scaler = MinMaxScaler()
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                        # Append the normalized data for this category
                        combined_data.append(df)
                        print(f"Processed: {file_path}")  # Debugging print

            # Combine all the dataframes into one for the train dataset
            if combined_data:
                final_train_df = pd.concat(combined_data, ignore_index=True)
                # Write the final combined dataframe to a single CSV file for all train data
                output_path = os.path.join(output_directory, output_file_name)
                final_train_df.to_csv(output_path, index=False)
                print(f"Final combined train CSV saved to: {output_path}")
            else:
                print(f"No valid data found for training. Skipping.")


# Main execution
# For training data
train_input_directory = r'C:\Git\EEG-Emotion-Analysis\eegEmotionAnalysis\NormalFeatureExtraction'  # Train dataset directory
output_directory = r'.\Normalized'  # Output directory for both train and test data
process_psd_summary(train_input_directory, output_directory, is_train=True)

# For test data
test_input_directory = r'C:\Git\EEG-Emotion-Analysis\eegEmotionAnalysis\StrokeFeatureExtraction'  # Test dataset directory
process_psd_summary(test_input_directory, output_directory, is_train=False)

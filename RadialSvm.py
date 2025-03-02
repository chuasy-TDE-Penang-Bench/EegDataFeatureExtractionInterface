import os
import pandas as pd
from sklearn.svm import SVC
import joblib

def train_svm(train_data_path, test_data_path, result_directory):
    """
    This function trains the SVM model using the provided training data and
    evaluates it using the test data. It also saves the predictions to the specified directory.
    """
    # Load the training data
    X_train, y_train = load_data(train_data_path, is_train=True)

    # Load the test data
    X_test, _ = load_data(test_data_path, is_train=False)

    # Train an SVM classifier with Radial Basis Function (RBF) kernel
    model = SVC(kernel='rbf')  # Changed to Radial Basis Function kernel
    model.fit(X_train, y_train)

    # Predict on the test data
    predictions = model.predict(X_test)

    # Combine the features and predictions into one DataFrame
    result_df = X_test.copy()  # Start with the test features
    result_df['Predictions'] = predictions  # Add the predictions as a new column

    # Save the combined results to a CSV in the specified result directory
    predictions_filename = os.path.join(result_directory, os.path.basename(test_data_path).replace('.csv', '_predictions.csv'))
    result_df.to_csv(predictions_filename, index=False)
    print(f"Predictions saved for {test_data_path} in {result_directory}")

    return model, predictions

def load_data(file_path, is_train=True):
    """
    This function loads the data from the specified CSV file, processes the
    feature columns and returns the features and labels.
    """
    # Load the CSV data
    df = pd.read_csv(file_path)

    if is_train:
        # For training data, separate features and labels
        X = df.drop(columns=['CLASS'])  # Features (exclude target column)
        y = df['CLASS']  # Target column (CLASS)
    else:
        # For test data, just return the features
        X = df
        y = None  # No labels for test data

    return X, y

def train_multiple_svms(train_data_path, test_data_directory, result_directory):
    """
    This function trains multiple SVMs for each test data file in the test data directory
    based on different severity and eyes conditions. Results will be saved to the result directory.
    """
    # Get the list of all test data files in the specified directory
    test_data_files = [f for f in os.listdir(test_data_directory) if f.endswith('.csv') and f != 'train_data.csv']

    # Loop over the test data files (based on severity and eyes)
    for test_data_file in test_data_files:
        test_data_path = os.path.join(test_data_directory, test_data_file)

        print(f"\nTraining Radial SVM for {test_data_file}...")
        # Train and test SVM for each condition
        model, predictions = train_svm(train_data_path, test_data_path, result_directory)

        # Optionally, save the model in the specified result directory
        model_filename = os.path.join(result_directory, f"{test_data_file}_radial_svm_model.pkl")
        joblib.dump(model, model_filename)
        print(f"Model saved for {test_data_file} in {result_directory}")

# Define paths
normalized_directory = "C:/Git/EEG-Emotion-Analysis/eegEmotionAnalysis/Normalized"
train_data_path = os.path.join(normalized_directory, "train_data.csv")
test_data_directory = normalized_directory  # The test data is in the same directory

# Define the result directory where the predictions and models will be saved
result_directory = "C:/Git/EEG-Emotion-Analysis/eegEmotionAnalysis/RadialResults"  # Changed to RadialResults

# Ensure the result directory exists
os.makedirs(result_directory, exist_ok=True)

# Train multiple SVMs for all the test data (excluding train_data.csv)
train_multiple_svms(train_data_path, test_data_directory, result_directory)

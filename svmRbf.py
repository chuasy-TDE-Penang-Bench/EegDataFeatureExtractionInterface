import os
import pandas as pd
from sklearn.svm import SVC
import joblib

def process(self):
    """
    Trains an SVM model with RBF kernel using self.train_data_path,
    predicts on self.test_data_path, and saves results.
    """

    # Initialize GUI elements
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    train_data_path = self.train_data_path.get()
    test_data_path = self.test_data_path.get()

    # Load training and test data
    X_train, y_train = load_data(train_data_path, is_train=True)
    X_test, _ = load_data(test_data_path, is_train=False)

    # Train SVM model with RBF kernel
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Get test data file name
    test_data_file_name = os.path.basename(test_data_path)

    # Save predictions
    result_df = X_test.copy()
    result_df['Predictions'] = predictions
    predictions_filename = os.path.join(output_dir, test_data_file_name.replace('.csv', '_predictions.csv'))
    result_df.to_csv(predictions_filename, index=False)
    self.log(f"Predictions saved to: {predictions_filename}")

    # Save model
    # model_filename = os.path.join(output_dir, "svm_model.pkl")
    model_filename = os.path.join(output_dir, test_data_file_name.replace('_test_data.csv', '_svm_model.pkl'))
    joblib.dump(model, model_filename)
    self.log(f"Model saved to: {model_filename}")


def load_data(file_path, is_train):
    """Loads and processes data from CSV."""
    df = pd.read_csv(file_path, header=0)
    if is_train:
        X = df.drop(columns=['CLASS'])
        y = df['CLASS']
    else:
        X = df
        y = None
    return X, y
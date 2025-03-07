import threading
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import time

def animate_progress(self):
    """Moves the progress bar smoothly from left to right repeatedly."""
    self.progress_bar["value"] = 0  # Start from 0
    while self.running:
        for value in range(0, 101, 2):  # Increase smoothly
            self.progress_bar["value"] = value
            self.root.update_idletasks()  # Force UI update
            time.sleep(0.02)  # Adjust speed for smooth motion
        self.progress_bar["value"] = 0  # Reset and start again

def process(self):
    """
    Trains an SVM model with hyperparameter tuning,
    predicts on test data, and saves results.
    """
    self.log("Starting processing...")
    self.running = True  # Start progress animation
    progress_thread = threading.Thread(target=animate_progress, args=(self,), daemon=True)
    progress_thread.start()

    try:
        output_dir = self.output_dir.get()
        train_data_path = self.train_data_path.get()
        test_data_path = self.test_data_path.get()
        model_file_path = self.model_file_path.get()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load training and test data
        X_train, y_train = load_data(train_data_path, is_train=True)
        X_test, _ = load_data(test_data_path, is_train=False)

        if not model_file_path:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'C': [1, 10],  # Reduce the number of values
                'gamma': ['scale', 0.01],  # Keep only two options
                'kernel': ['rbf', 'poly']  # Test only two kernels
            }
            # param_grid = {
            #     'C': [0.1, 1, 10, 100],
            #     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            #     'kernel': ['rbf', 'poly', 'sigmoid']  # Trying different kernels
            # }
            grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            self.log(f"Best hyperparameters: {best_params}")

            # Save the best model
            model_filename = os.path.join(output_dir, 'svm_model.pkl')
            joblib.dump(model, model_filename)
            self.log(f"Saved model: {model_filename}")

        else:
            # Load the trained model
            model = joblib.load(model_file_path)
            self.log(f"Loaded model: {os.path.normpath(model_file_path)}")
            self.log(f"Best hyperparameters: {model.get_params()}")

        # Predict on test data
        predictions = model.predict(X_test)

        # Save predictions
        test_data_file_name = os.path.basename(test_data_path)
        result_df = pd.DataFrame(X_test)  # X_test retains original features
        result_df['Predictions'] = predictions
        predictions_filename = os.path.join(output_dir, test_data_file_name.replace('.csv', '_predictions.csv'))
        result_df.to_csv(predictions_filename, index=False)
        self.log(f"Predictions saved to: {predictions_filename}")

    finally:
        # Stop the progress bar animation after processing is complete
        self.running = False
        self.progress_bar["value"] = 100  # Set progress bar to full when done



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

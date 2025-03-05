import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib


def process(self):
    """
    Trains an SVM model with hyperparameter tuning,
    predicts on test data, and saves results.
    """
    self.log("Starting processing...")
    output_dir = self.output_dir.get()
    train_data_path = self.train_data_path.get()
    test_data_path = self.test_data_path.get()

    # Load training and test data
    X_train, y_train = load_data(train_data_path, is_train=True)
    X_test, _ = load_data(test_data_path, is_train=False)

    # # Hyperparameter tuning with GridSearchCV
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'poly', 'sigmoid']  # Trying different kernels
    # }
    # grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)
    # best_model = grid_search.best_estimator_

    # Load the trained model
    model_filename = f"best_svm_model.pkl"
    model_path = os.path.join(output_dir, model_filename)
    best_model = joblib.load(model_path)
    print(f"Loaded file: {os.path.normpath(model_path)}")

    # Predict on test data
    predictions = best_model.predict(X_test)

    # Save predictions
    test_data_file_name = os.path.basename(test_data_path)
    result_df = pd.DataFrame(X_test)  # X_test retains original features
    result_df['Predictions'] = predictions
    predictions_filename = os.path.join(output_dir, test_data_file_name.replace('.csv', '_predictions.csv'))
    result_df.to_csv(predictions_filename, index=False)
    self.log(f"Predictions saved to: {predictions_filename}")

    # Save the best model
    # model_filename = os.path.join(output_dir, test_data_file_name.replace('_test_data.csv', '_svm_model.pkl'))
    # joblib.dump(best_model, model_filename)
    # self.log(f"Best Model saved to: {model_filename}")


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

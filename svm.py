import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path, is_train=True):
    """
    This function loads the data from the specified CSV file, processes the
    feature columns, and returns the features and labels.
    """
    # Load the CSV data
    df = pd.read_csv(file_path)

    if is_train:
        # For training data, separate features and labels
        X = df.drop(columns=['CLASS'])  # Features (exclude target column)
        y = df['CLASS']  # Target column (CLASS)

        # Convert string labels to numeric labels using LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)  # Transform the labels to numeric

    else:
        # For test data, just return the features
        X = df
        y = None  # No labels for test data

    return X, y


def visualize_svm_decision_boundaries(train_data_path):
    """
    Visualize decision boundaries for both Linear and RBF SVM using PCA
    for dimensionality reduction (to 2D).
    """
    # Load training data
    X, y = load_data(train_data_path, is_train=True)

    # Standardize the data (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce data to 2D using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # Train Linear SVM
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X_train, y_train)

    # Train RBF SVM
    rbf_svm = SVC(kernel='rbf')
    rbf_svm.fit(X_train, y_train)

    # Create a mesh grid for plotting decision boundaries
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict on the grid points for Linear SVM
    Z_linear = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_linear = Z_linear.reshape(xx.shape)

    # Predict on the grid points for RBF SVM
    Z_rbf = rbf_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_rbf = Z_rbf.reshape(xx.shape)

    # Class labels in their string form for the legend
    class_labels = ['happy', 'sad', 'neutral', 'fear']

    # Create a color map for the classes
    colors = ['#FF6347', '#4682B4', '#3CB371', '#FFD700']  # Assign specific colors to each class

    # Plot Linear SVM decision boundary
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # Left side plot
    plt.contourf(xx, yy, Z_linear, alpha=0.75, cmap=plt.cm.RdBu)

    # Plotting points with distinct colors and markers
    for i, class_label in enumerate(class_labels):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=colors[i], label=class_label, edgecolors='k', marker='o',
                    s=60)

    plt.title('Linear SVM')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    # Plot RBF SVM decision boundary
    plt.subplot(1, 2, 2)  # Right side plot
    plt.contourf(xx, yy, Z_rbf, alpha=0.75, cmap=plt.cm.RdBu)

    # Plotting points with distinct colors and markers
    for i, class_label in enumerate(class_labels):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=colors[i], label=class_label, edgecolors='k', marker='o',
                    s=60)

    plt.title('RBF SVM')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage
normalized_directory = "C:/Git/EEG-Emotion-Analysis/eegEmotionAnalysis/Normalized"
train_data_path = os.path.join(normalized_directory, "train_data.csv")

# Visualize decision boundaries for Linear and RBF SVM
visualize_svm_decision_boundaries(train_data_path)

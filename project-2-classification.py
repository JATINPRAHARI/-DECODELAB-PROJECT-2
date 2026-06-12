import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassificationPipeline:
    """Iris Dataset Classification using KNN."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, k_neighbors: int = 5):
        self.test_size = test_size
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k_neighbors)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.confusion_mat = None
        self.accuracy = None
        self.f1 = None
        
        self.feature_names = None
        self.target_names = None

    def load_data(self):
        """Load Iris dataset."""
        print("\n" + "="*60)
        print("STEP 1: Loading Iris Dataset")
        print("="*60)
        
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {list(self.target_names)}")
        return X, y

    def normalize_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        print("\nSTEP 2: Feature Scaling")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Features scaled successfully")
        return X_train_scaled, X_test_scaled

    def split_data(self, X, y):
        """Split data into train and test sets."""
        print("\nSTEP 3: Train-Test Split")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y, shuffle=True
        )
        
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"✓ Split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    def train_model(self):
        """Train KNN model."""
        print("\nSTEP 4: Training KNN Model")
        print(f"Using K = {self.k_neighbors} neighbors")
        
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained successfully")

    def make_predictions(self):
        """Make predictions on test set."""
        print("\nSTEP 5: Making Predictions")
        self.y_pred = self.model.predict(self.X_test)
        print(f"✓ Predictions done for {len(self.y_pred)} samples")

    def evaluate_model(self):
        """Evaluate model performance."""
        print("\nSTEP 6: Model Evaluation")
        
        self.confusion_mat = confusion_matrix(self.y_test, self.y_pred)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        print(f"\nAccuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"F1 Score: {self.f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.target_names))

    def visualize_confusion_matrix(self):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - KNN (K={self.k_neighbors})\nAccuracy: {self.accuracy:.2%}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=200)
        print("✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()

    def run_pipeline(self):
        """Run the complete classification pipeline."""
        print("\n" + "="*70)
        print("🌸 IRIS CLASSIFICATION PIPELINE - PROJECT 2")
        print("="*70)
        
        X, y = self.load_data()
        self.split_data(X, y)
        self.train_model()
        self.make_predictions()
        self.evaluate_model()
        self.visualize_confusion_matrix()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)

def main():
    pipeline = IrisClassificationPipeline(k_neighbors=5)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

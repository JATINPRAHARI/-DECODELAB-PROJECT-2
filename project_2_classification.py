"""
PROJECT 2: DATA CLASSIFICATION USING AI 🌸
DecodeLabs Industrial Training Kit - Batch 2026
Artificial Intelligence: Predictive Phase

This project demonstrates supervised learning through the fundamental
pipeline: Data Loading → Feature Scaling → Train-Test Split → 
Algorithm Application → Output Validation

Key Concepts:
- Supervised Learning: Learning from labeled historical data
- Feature Scaling: Normalizing data for better algorithm performance
- Train-Test Split: Separating data to prevent overfitting
- K-Nearest Neighbors: Simple proximity-based classification
- Confusion Matrix: Understanding True Positives, False Positives, etc.
- F1 Score: Harmonic mean of precision and recall
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


class IrisClassificationPipeline:
    """
    The Master Blueprint: IPO Framework
    
    INPUT (Iris Dataset) → PROCESS (ML Pipeline) → OUTPUT (Validation Metrics)
    
    This class encapsulates the complete supervised learning workflow.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, k_neighbors: int = 5):
        """
        Initialize the classification pipeline.
        
        Args:
            test_size: Proportion of data for testing (default 20%)
            random_state: Seed for reproducibility
            k_neighbors: Number of neighbors for KNN algorithm
        """
        self.test_size = test_size
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Model components
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k_neighbors)
        
        # Predictions and metrics
        self.y_pred = None
        self.confusion_mat = None
        self.f1 = None
        self.accuracy = None
        
        # Dataset info
        self.iris = None
        self.feature_names = None
        self.target_names = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        STEP 1: INPUT - Load Iris Dataset
        
        Raw Material: The Iris Benchmark
        - Samples: 150 (Balanced)
        - Classes: 3 (Setosa, Versicolor, Virginica)
        - Dimensions: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
        - Unit: Centimeters
        
        Returns:
            Tuple of (features, labels)
        """
        print("\n" + "="*70)
        print("STEP 1: INPUT - Loading Iris Dataset")
        print("="*70)
        
        self.iris = load_iris()
        X = self.iris.data
        y = self.iris.target
        
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  • Samples: {X.shape[0]} (Balanced)")
        print(f"  • Classes: {len(self.target_names)} {list(self.target_names)}")
        print(f"  • Dimensions (Features): {X.shape[1]}")
        print(f"\n  Feature Names:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"    {i}. {name}")
        
        return X, y
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        STEP 2: PROCESS - Feature Scaling (The Gatekeeper Rule)
        
        Raw Data Issue: Features have different scales
        - Sepal Length: 4.3 - 7.9 cm
        - Sepal Width: 2.0 - 4.4 cm
        - Petal Length: 1.0 - 6.9 cm
        - Petal Width: 0.1 - 2.5 cm
        
        Solution: StandardScaler normalizes to Mean=0, Variance=1
        This prevents features with larger scales from dominating KNN.
        
        Args:
            X_train: Training features (raw)
            X_test: Testing features (raw)
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        print("\n" + "="*70)
        print("STEP 2: PROCESS - Feature Scaling (StandardScaler)")
        print("="*70)
        
        print("\n📊 Raw Data Statistics (Before Scaling):")
        print(f"\n  Training Set Shape: {X_train.shape}")
        for i, name in enumerate(self.feature_names):
            print(f"  {name}:")
            print(f"    Range: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}] cm")
            print(f"    Mean: {X_train[:, i].mean():.2f}, Std: {X_train[:, i].std():.2f}")
        
        # Fit scaler on training data only (prevents data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n✓ Scaling applied: StandardScaler (Mean=0, Variance=1)")
        print("\n📊 Scaled Data Statistics (After Scaling):")
        for i, name in enumerate(self.feature_names):
            print(f"  {name}:")
            print(f"    Mean: {X_train_scaled[:, i].mean():.6f}, Std: {X_train_scaled[:, i].std():.6f}")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        STEP 3: PROCESS - Structural Integrity (Train-Test Split)
        
        Why Split?
        - Training Set: Teach the model patterns
        - Testing Set: Validate on unseen data (prevents overfitting)
        
        Split Ratio: 80% train, 20% test
        Shuffle: Randomize before splitting to remove order bias
        
        Args:
            X: All features
            y: All labels
        """
        print("\n" + "="*70)
        print("STEP 3: PROCESS - Train-Test Split")
        print("="*70)
        
        # Split with shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=y  # Maintain class distribution
        )
        
        print(f"\n✓ Data split successfully (stratified):")
        print(f"  • Training Set: {X_train.shape[0]} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"  • Testing Set: {X_test.shape[0]} samples ({self.test_size*100:.0f}%)")
        
        # Show class distribution
        print(f"\n  Class Distribution in Training Set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"    {self.target_names[class_idx]}: {count} samples")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        # Store in instance
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
    
    def train_model(self) -> None:
        """
        STEP 4: PROCESS - Algorithm Application (K-Nearest Neighbors)
        
        KNN Principle: Similar things exist in close proximity
        - Find K nearest neighbors in training data
        - Assign class by majority vote
        
        Why KNN?
        - Simple and interpretable (white box)
        - Non-parametric (no assumptions)
        - Good baseline for classification
        
        Our Choice: K=5 (optimal for Iris dataset)
        """
        print("\n" + "="*70)
        print("STEP 4: PROCESS - Algorithm Training (K-Nearest Neighbors)")
        print("="*70)
        
        print(f"\n🤖 Training KNN Classifier...")
        print(f"  • Algorithm: K-Nearest Neighbors")
        print(f"  • K value: {self.k_neighbors} neighbors")
        print(f"  • Distance metric: Euclidean")
        print(f"  • Training samples: {self.X_train.shape[0]}")
        
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Model trained successfully!")
        print(f"  The model memorized the training data patterns.")
    
    def make_predictions(self) -> None:
        """
        STEP 5: PROCESS - Logic Application (Prediction)
        
        Apply learned patterns to test set.
        The model looks up K nearest neighbors for each test sample
        and assigns the majority class.
        """
        print("\n" + "="*70)
        print("STEP 5: PROCESS - Making Predictions")
        print("="*70)
        
        print(f"\n🔮 Generating predictions on test set...")
        self.y_pred = self.model.predict(self.X_test)
        
        print(f"✓ Predictions generated for {len(self.y_pred)} test samples")
    
    def evaluate_model(self) -> None:
        """
        STEP 6: OUTPUT - Output Validation (Confusion Matrix & F1 Score)
        
        Confusion Matrix:
        - TP (True Positive): Correctly classified as positive
        - FP (False Positive): Incorrectly classified as positive
        - FN (False Negative): Missed detection
        - TN (True Negative): Correctly classified as negative
        
        F1 Score (Harmonic Mean):
        - Balance between Precision and Recall
        - Range: 0 to 1 (1 = perfect)
        
        Accuracy:
        - Overall correctness: (TP + TN) / Total
        """
        print("\n" + "="*70)
        print("STEP 6: OUTPUT - Model Evaluation")
        print("="*70)
        
        # Calculate metrics
        self.confusion_mat = confusion_matrix(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        
        # Display Confusion Matrix
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"\n  Predicted ↓ / Actual →")
        
        # Header
        header = "        "
        for i, name in enumerate(self.target_names):
            header += f"{name[:10]:>15}"
        print(header)
        
        # Rows
        for i, name in enumerate(self.target_names):
            row = f"{name[:10]:>7}"
            for j in range(len(self.target_names)):
                row += f"{self.confusion_mat[i][j]:>15}"
            print(row)
        
        # Display Metrics
        print(f"\n\n📈 PERFORMANCE METRICS:")
        print(f"  • Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"  • F1 Score (Weighted): {self.f1:.4f}")
        print(f"  • Model Performance: ", end="")
        
        if self.accuracy >= 0.95:
            print("🟢 EXCELLENT")
        elif self.accuracy >= 0.85:
            print("🟢 VERY GOOD")
        elif self.accuracy >= 0.75:
            print("🟡 GOOD")
        else:
            print("🔴 NEEDS IMPROVEMENT")
        
        # Detailed Classification Report
        print(f"\n\n📋 DETAILED CLASSIFICATION REPORT:")
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.target_names,
            digits=4
        )
        print(report)
    
    def visualize_confusion_matrix(self) -> None:
        """
        Create a heatmap visualization of the confusion matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.target_names,
            yticklabels=self.target_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - KNN (K={self.k_neighbors})\nAccuracy: {self.accuracy:.2%}', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('/home/claude/confusion_matrix.png', dpi=100, bbox_inches='tight')
        print("\n✓ Confusion matrix visualization saved as 'confusion_matrix.png'")
        plt.close()
    
    def run_pipeline(self) -> None:
        """
        Execute the complete IPO pipeline: INPUT → PROCESS → OUTPUT
        """
        print("\n" + "="*80)
        print(" "*15 + "🌸 IRIS CLASSIFICATION PIPELINE - PROJECT 2 🌸")
        print(" "*10 + "Supervised Learning: Data Classification Using AI")
        print("="*80)
        
        try:
            # INPUT Phase
            X, y = self.load_data()
            
            # PROCESS Phase
            self.split_data(X, y)
            self.train_model()
            self.make_predictions()
            
            # OUTPUT Phase
            self.evaluate_model()
            self.visualize_confusion_matrix()
            
            print("\n" + "="*80)
            print("✓ PIPELINE EXECUTION COMPLETE")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during pipeline execution: {e}")
            raise


def analyze_k_values():
    """
    Bonus: Analyze different K values to find optimal K.
    
    Demonstrates the concept of "TUNING THE ENGINE: CHOOSING K"
    """
    print("\n" + "="*80)
    print("BONUS: K-VALUE ANALYSIS (Tuning the Engine)")
    print("="*80)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Test different K values
    k_values = range(1, 31)
    accuracies = []
    f1_scores = []
    
    print("\nTesting K values from 1 to 30...\n")
    print(f"{'K':>3} | {'Accuracy':>10} | {'F1 Score':>10}")
    print("-" * 30)
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        if k in [1, 3, 5, 7, 10, 15, 20, 25, 30]:
            print(f"{k:3d} | {acc:10.4f} | {f1:10.4f}")
    
    # Find optimal K
    optimal_k = k_values[np.argmax(accuracies)]
    max_accuracy = max(accuracies)
    
    print("\n" + "-"*30)
    print(f"🏆 Optimal K: {optimal_k} (Accuracy: {max_accuracy:.4f})")
    print("="*80 + "\n")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('K Value', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Accuracy vs K Value', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, f1_scores, marker='s', linestyle='-', linewidth=2, markersize=6, color='orange')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('K Value', fontsize=11)
    plt.ylabel('F1 Score', fontsize=11)
    plt.title('F1 Score vs K Value', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/k_value_analysis.png', dpi=100, bbox_inches='tight')
    print("✓ K-value analysis visualization saved as 'k_value_analysis.png'\n")
    plt.close()


def main():
    """
    Main execution function.
    """
    # Run the main pipeline
    pipeline = IrisClassificationPipeline(k_neighbors=5)
    pipeline.run_pipeline()
    
    # Bonus: Analyze K values
    analyze_k_values()


if __name__ == "__main__":
    main()

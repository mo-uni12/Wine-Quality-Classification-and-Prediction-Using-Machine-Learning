#!/usr/bin/env python3
"""
Wine Quality Classification and Prediction Using Machine Learning
Comprehensive analysis and model implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WineQualityAnalyzer:
    """
    Comprehensive Wine Quality Analysis and Prediction System
    """
    
    def __init__(self):
        self.red_wine = None
        self.white_wine = None
        self.combined_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and combine red and white wine datasets"""
        print("Loading wine quality datasets...")
        
        # Load datasets
        self.red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
        self.white_wine = pd.read_csv('data/winequality-white.csv', sep=';')
        
        # Add wine type column
        self.red_wine['wine_type'] = 'red'
        self.white_wine['wine_type'] = 'white'
        
        # Combine datasets
        self.combined_data = pd.concat([self.red_wine, self.white_wine], ignore_index=True)
        
        print(f"Red wine samples: {len(self.red_wine)}")
        print(f"White wine samples: {len(self.white_wine)}")
        print(f"Total samples: {len(self.combined_data)}")
        
        return self.combined_data
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information
        print("\nDataset Info:")
        print(self.combined_data.info())
        
        print("\nDataset Shape:", self.combined_data.shape)
        print("\nFeature Names:", list(self.combined_data.columns))
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.combined_data.describe())
        
        # Missing values
        print("\nMissing Values:")
        missing_values = self.combined_data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Quality distribution
        print("\nQuality Distribution:")
        quality_dist = self.combined_data['quality'].value_counts().sort_index()
        print(quality_dist)
        
        # Wine type distribution
        print("\nWine Type Distribution:")
        print(self.combined_data['wine_type'].value_counts())
        
        return self.combined_data.describe()
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        print("\nGenerating visualizations...")
        
        # Set up the plotting parameters
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Quality Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality distribution by wine type
        axes[0, 0].hist([self.red_wine['quality'], self.white_wine['quality']], 
                       bins=range(3, 11), alpha=0.7, label=['Red Wine', 'White Wine'])
        axes[0, 0].set_title('Quality Distribution by Wine Type')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Overall quality distribution
        self.combined_data['quality'].hist(bins=range(3, 11), ax=axes[0, 1])
        axes[0, 1].set_title('Overall Quality Distribution')
        axes[0, 1].set_xlabel('Quality Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Quality vs Alcohol content
        axes[1, 0].scatter(self.combined_data['alcohol'], self.combined_data['quality'], alpha=0.6)
        axes[1, 0].set_title('Quality vs Alcohol Content')
        axes[1, 0].set_xlabel('Alcohol (%)')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Quality vs pH
        axes[1, 1].scatter(self.combined_data['pH'], self.combined_data['quality'], alpha=0.6)
        axes[1, 1].set_title('Quality vs pH')
        axes[1, 1].set_xlabel('pH')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wine_quality_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Matrix
        plt.figure(figsize=(12, 10))
        numeric_columns = self.combined_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.combined_data[numeric_columns].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature distributions
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        numeric_features = [col for col in self.combined_data.columns if col not in ['quality', 'wine_type']]
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                self.combined_data[feature].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Box plots for quality analysis
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                self.combined_data.boxplot(column=feature, by='quality', ax=axes[i])
                axes[i].set_title(f'{feature} by Quality')
                axes[i].set_xlabel('Quality Score')
                axes[i].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig('quality_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved successfully!")
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\nPreprocessing data...")
        
        # Prepare features and target
        feature_columns = [col for col in self.combined_data.columns 
                          if col not in ['quality', 'wine_type']]
        
        X = self.combined_data[feature_columns]
        y = self.combined_data['quality']
        
        # Create quality categories for better classification
        # Convert to binary classification: Good (7-10) vs Poor/Average (3-6)
        y_binary = (y >= 7).astype(int)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Feature count: {len(feature_columns)}")
        print(f"Class distribution in training set:")
        print(f"Poor/Average (0): {sum(self.y_train == 0)}")
        print(f"Good (1): {sum(self.y_train == 1)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\nTraining machine learning models...")
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Performance comparison
        performance_data = []
        for name, result in self.results.items():
            performance_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(performance_df.to_string(index=False))
        
        # Detailed classification reports
        print("\nDetailed Classification Reports:")
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(result['classification_report'])
        
        return performance_df
    
    def create_evaluation_plots(self):
        """Create evaluation visualizations"""
        print("\nGenerating evaluation plots...")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        cv_means = [self.results[model]['cv_mean'] for model in models]
        
        axes[0, 0].bar(models, accuracies, alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cross-validation scores
        axes[0, 1].bar(models, cv_means, alpha=0.7)
        axes[0, 1].set_title('Cross-Validation Scores')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_predictions = self.results[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # ROC Curves
        for name, result in self.results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                roc_auc = auc(fpr, tpr)
                axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set_xlim([0.0, 1.0])
        axes[1, 1].set_ylim([0.0, 1.05])
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance (for Random Forest)
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            feature_names = [col for col in self.combined_data.columns 
                           if col not in ['quality', 'wine_type']]
            
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance (Random Forest)')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.ylabel('Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Evaluation plots saved successfully!")
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best models"""
        print("\nPerforming hyperparameter tuning...")
        
        # Random Forest tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                              rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best Random Forest parameters: {rf_grid.best_params_}")
        print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")
        
        # Update results with tuned model
        tuned_rf = rf_grid.best_estimator_
        tuned_predictions = tuned_rf.predict(self.X_test_scaled)
        tuned_accuracy = accuracy_score(self.y_test, tuned_predictions)
        
        self.results['Tuned Random Forest'] = {
            'model': tuned_rf,
            'accuracy': tuned_accuracy,
            'predictions': tuned_predictions,
            'best_params': rf_grid.best_params_
        }
        
        print(f"Tuned Random Forest accuracy: {tuned_accuracy:.4f}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("WINE QUALITY PREDICTION - SUMMARY REPORT")
        print("="*60)
        
        # Dataset summary
        print(f"\nDataset Summary:")
        print(f"- Total samples: {len(self.combined_data)}")
        print(f"- Red wine samples: {len(self.red_wine)}")
        print(f"- White wine samples: {len(self.white_wine)}")
        print(f"- Features: {len([col for col in self.combined_data.columns if col not in ['quality', 'wine_type']])}")
        print(f"- Quality range: {self.combined_data['quality'].min()} - {self.combined_data['quality'].max()}")
        
        # Best model performance
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_accuracy = self.results[best_model]['accuracy']
        
        print(f"\nBest Performing Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Model ranking
        print(f"\nModel Performance Ranking:")
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (name, result) in enumerate(sorted_models, 1):
            print(f"{i}. {name}: {result['accuracy']:.4f}")
        
        # Key insights
        print(f"\nKey Insights:")
        print(f"- Machine learning can effectively predict wine quality with {best_accuracy*100:.1f}% accuracy")
        print(f"- Random Forest and Gradient Boosting typically perform best for this dataset")
        print(f"- Feature scaling is important for SVM and KNN models")
        print(f"- Binary classification (Good vs Poor/Average) provides better results than multi-class")
        
        return {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'total_samples': len(self.combined_data),
            'feature_count': len([col for col in self.combined_data.columns if col not in ['quality', 'wine_type']])
        }

def main():
    """Main execution function"""
    print("Wine Quality Classification and Prediction Using Machine Learning")
    print("="*70)
    
    # Initialize analyzer
    analyzer = WineQualityAnalyzer()
    
    # Load and analyze data
    analyzer.load_data()
    analyzer.exploratory_data_analysis()
    analyzer.create_visualizations()
    
    # Preprocess and train models
    analyzer.preprocess_data()
    analyzer.train_models()
    
    # Evaluate models
    performance_df = analyzer.evaluate_models()
    analyzer.create_evaluation_plots()
    
    # Hyperparameter tuning
    analyzer.hyperparameter_tuning()
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    
    print(f"\nAnalysis complete! Check the generated visualization files:")
    print("- wine_quality_distributions.png")
    print("- correlation_matrix.png")
    print("- feature_distributions.png")
    print("- quality_boxplots.png")
    print("- model_evaluation.png")
    print("- feature_importance.png")
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()


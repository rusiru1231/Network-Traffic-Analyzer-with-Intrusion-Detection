"""
Model training script for intrusion detection system.
Train and evaluate ML models on CICIDS 2017 or NSL-KDD datasets.
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import DataPreprocessor
from models.intrusion_detector import IntrusionDetector
from utils.logger import setup_logging
from utils.config_manager import ConfigManager


class ModelTrainer:
    """Train and evaluate intrusion detection models."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize model trainer."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)
        
        self.preprocessor = None
        self.models = {}
        self.results = {}
    
    def train_models(self, data_path: str, dataset_name: str = 'cicids2017'):
        """
        Train multiple models and compare performance.
        
        Args:
            data_path: Path to dataset
            dataset_name: Name of dataset ('cicids2017' or 'nsl_kdd')
        """
        try:
            self.logger.info(f"Starting model training for {dataset_name} dataset...")
            
            # 1. Load and preprocess data
            self.preprocessor = DataPreprocessor(dataset_name)
            df = self.preprocessor.load_data(data_path)
            datasets = self.preprocessor.preprocess_data(df)
            
            # 2. Define models to train
            models_config = {
                'random_forest': {
                    'model': RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'name': 'Random Forest'
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    ),
                    'name': 'Gradient Boosting'
                },
                'neural_network': {
                    'model': MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        learning_rate_init=0.001,
                        random_state=42
                    ),
                    'name': 'Neural Network'
                },
                'svm': {
                    'model': SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True,
                        random_state=42
                    ),
                    'name': 'Support Vector Machine'
                }
            }
            
            # 3. Train and evaluate each model
            for model_key, model_config in models_config.items():
                self.logger.info(f"Training {model_config['name']}...")
                
                model = model_config['model']
                
                # Train model
                model.fit(datasets['X_train'], datasets['y_train'])
                
                # Evaluate model
                results = self._evaluate_model(
                    model, 
                    datasets,
                    model_config['name']
                )
                
                # Store model and results
                self.models[model_key] = model
                self.results[model_key] = results
                
                self.logger.info(f"{model_config['name']} - Accuracy: {results['accuracy']:.3f}, "
                               f"Precision: {results['precision']:.3f}, "
                               f"Recall: {results['recall']:.3f}")
            
            # 4. Select best model
            best_model_key = self._select_best_model()
            self.logger.info(f"Best model: {models_config[best_model_key]['name']}")
            
            # 5. Save best model
            self._save_best_model(best_model_key, datasets)
            
            # 6. Generate reports
            self._generate_training_report(datasets)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def _evaluate_model(self, model, datasets: dict, model_name: str) -> dict:
        """Evaluate model performance on all datasets."""
        results = {'model_name': model_name}
        
        # Evaluate on training set
        train_pred = model.predict(datasets['X_train'])
        train_proba = model.predict_proba(datasets['X_train'])[:, 1] if hasattr(model, 'predict_proba') else None
        
        results['train_accuracy'] = (train_pred == datasets['y_train']).mean()
        if train_proba is not None:
            results['train_auc'] = roc_auc_score(datasets['y_train'], train_proba)
        
        # Evaluate on validation set
        val_pred = model.predict(datasets['X_val'])
        val_proba = model.predict_proba(datasets['X_val'])[:, 1] if hasattr(model, 'predict_proba') else None
        
        results['val_accuracy'] = (val_pred == datasets['y_val']).mean()
        if val_proba is not None:
            results['val_auc'] = roc_auc_score(datasets['y_val'], val_proba)
        
        # Evaluate on test set
        test_pred = model.predict(datasets['X_test'])
        test_proba = model.predict_proba(datasets['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results['accuracy'] = accuracy_score(datasets['y_test'], test_pred)
        results['precision'] = precision_score(datasets['y_test'], test_pred, average='weighted')
        results['recall'] = recall_score(datasets['y_test'], test_pred, average='weighted')
        results['f1_score'] = f1_score(datasets['y_test'], test_pred, average='weighted')
        
        if test_proba is not None:
            results['auc'] = roc_auc_score(datasets['y_test'], test_proba)
        
        # Detailed classification report
        results['classification_report'] = classification_report(
            datasets['y_test'], test_pred, output_dict=True
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(datasets['y_test'], test_pred)
        
        return results
    
    def _select_best_model(self) -> str:
        """Select best model based on F1-score."""
        best_model = None
        best_score = -1
        
        for model_key, results in self.results.items():
            f1_score = results['f1_score']
            if f1_score > best_score:
                best_score = f1_score
                best_model = model_key
        
        return best_model
    
    def _save_best_model(self, best_model_key: str, datasets: dict):
        """Save the best performing model."""
        try:
            # Create intrusion detector with best model
            detector = IntrusionDetector()
            detector.model = self.models[best_model_key]
            detector.scaler = self.preprocessor.scaler
            detector.feature_names = self.preprocessor.feature_names
            
            # Add metadata
            detector.model_metadata = {
                'model_type': best_model_key,
                'dataset': self.preprocessor.dataset_name,
                'training_date': pd.Timestamp.now().isoformat(),
                'performance': self.results[best_model_key],
                'n_features': datasets['X_train'].shape[1]
            }
            
            # Save model
            model_path = self.config['data']['model_save_path']
            detector.save_model(model_path)
            
            # Save preprocessor
            preprocessor_path = model_path.replace('.joblib', '_preprocessor.joblib')
            self.preprocessor.save_preprocessor(preprocessor_path)
            
            self.logger.info(f"Best model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def _generate_training_report(self, datasets: dict):
        """Generate comprehensive training report."""
        try:
            report_path = Path("reports")
            report_path.mkdir(exist_ok=True)
            
            # 1. Model comparison plot
            self._plot_model_comparison()
            
            # 2. Confusion matrices
            self._plot_confusion_matrices()
            
            # 3. Feature importance (for tree-based models)
            self._plot_feature_importance()
            
            # 4. Training summary
            self._save_training_summary(datasets)
            
            self.logger.info("Training report generated in 'reports' directory")
            
        except Exception as e:
            self.logger.error(f"Error generating training report: {e}")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = [self.results[key]['model_name'] for key in self.results.keys()]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[key][metric] for key in self.results.keys()]
            
            axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_key, results) in enumerate(self.results.items()):
            if i >= 4:  # Only plot first 4 models
                break
                
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                cbar=False
            )
            
            axes[i].set_title(f'{results["model_name"]} Confusion Matrix')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        # Hide empty subplots
        for i in range(len(self.results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('reports/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        try:
            # Get feature importance from Random Forest model
            if 'random_forest' in self.models:
                model = self.models['random_forest']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = self.preprocessor.feature_names
                    
                    if feature_names and len(feature_names) == len(importances):
                        # Create feature importance DataFrame
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False).head(20)
                        
                        # Plot
                        plt.figure(figsize=(10, 8))
                        plt.barh(range(len(importance_df)), importance_df['importance'])
                        plt.yticks(range(len(importance_df)), importance_df['feature'])
                        plt.xlabel('Feature Importance')
                        plt.title('Top 20 Feature Importances (Random Forest)')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
                        plt.close()
        
        except Exception as e:
            self.logger.warning(f"Could not generate feature importance plot: {e}")
    
    def _save_training_summary(self, datasets: dict):
        """Save training summary to file."""
        summary = {
            'dataset_info': {
                'name': self.preprocessor.dataset_name,
                'original_shape': datasets['metadata']['original_shape'],
                'processed_shape': datasets['metadata']['processed_shape'],
                'class_distribution': datasets['metadata']['class_distribution']
            },
            'model_results': {}
        }
        
        for model_key, results in self.results.items():
            summary['model_results'][model_key] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'auc': results.get('auc', 'N/A')
            }
        
        # Save as JSON
        import json
        with open('reports/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train intrusion detection models")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to dataset file or directory"
    )
    parser.add_argument(
        "--dataset",
        choices=['cicids2017', 'nsl_kdd'],
        default='cicids2017',
        help="Dataset name"
    )
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)
        
        # Train models
        results = trainer.train_models(args.data_path, args.dataset)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        
        for model_key, result in results.items():
            print(f"\n{result['model_name']}:")
            print(f"  Accuracy:  {result['accuracy']:.3f}")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall:    {result['recall']:.3f}")
            print(f"  F1-Score:  {result['f1_score']:.3f}")
        
        print(f"\nBest model saved to: {trainer.config['data']['model_save_path']}")
        print(f"Training report saved to: reports/")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
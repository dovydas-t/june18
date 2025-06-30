# documentation/parameter_docs.py - Parameter Documentation
"""
Comprehensive documentation for all machine learning model parameters.
Provides detailed explanations, mathematical foundations, and practical guidance.
"""

from typing import Dict, Any


class ParameterDocumentation:
    """
    Comprehensive documentation for all machine learning model parameters.
    Provides detailed explanations, mathematical foundations, and practical guidance.
    """
    
    @staticmethod
    def get_model_docs() -> Dict[str, Dict[str, Any]]:
        """
        Returns comprehensive documentation for all supported models.
        
        Returns:
            Dict containing model documentation with parameters, descriptions, and examples
        """
        return {
            "KNN": {
                "name": "K-Nearest Neighbors",
                "description": "Instance-based learning algorithm that classifies data points based on the majority class of their k nearest neighbors in feature space.",
                "mathematical_foundation": "Distance-based: d(x,y) = √(Σ(xi-yi)²) for Euclidean distance",
                "use_cases": ["Small datasets", "Non-linear patterns", "Recommendation systems", "Anomaly detection"],
                "pros": ["Simple to understand", "No assumptions about data", "Works with non-linear data", "Good for small datasets"],
                "cons": ["Computationally expensive", "Sensitive to irrelevant features", "Poor performance on high-dimensional data", "Memory intensive"],
                "parameters": {
                    "n_neighbors": {
                        "description": "Number of neighbors to consider for prediction",
                        "range": "1 to sqrt(n_samples), typically 3-15",
                        "effect": "Lower values = more complex decision boundary (overfitting), Higher values = smoother boundary (underfitting)",
                        "tuning_tips": "Use odd numbers for binary classification to avoid ties. Start with sqrt(n_samples)."
                    },
                    "weights": {
                        "description": "Weight function for neighbor contributions",
                        "options": {"uniform": "All neighbors weighted equally", "distance": "Closer neighbors have more influence"},
                        "effect": "Distance weighting reduces noise from far neighbors",
                        "recommendation": "Use 'distance' for noisy data, 'uniform' for clean data"
                    },
                    "metric": {
                        "description": "Distance metric for finding neighbors",
                        "options": {"euclidean": "Standard L2 distance", "manhattan": "L1 distance (city block)", "chebyshev": "L∞ distance"},
                        "effect": "Euclidean works well for continuous features, Manhattan for categorical/sparse data",
                        "recommendation": "Start with Euclidean, try Manhattan if features have different scales"
                    }
                },
                "preprocessing_requirements": ["Feature scaling essential", "Handle missing values", "Consider dimensionality reduction"],
                "complexity": "O(n*d) for prediction where n=samples, d=dimensions"
            },
            
            "SVM": {
                "name": "Support Vector Machine",
                "description": "Finds optimal hyperplane to separate classes or predict continuous values by maximizing margin between support vectors.",
                "mathematical_foundation": "Optimization: min ½||w||² + C∑ξi subject to yi(w·xi + b) ≥ 1 - ξi",
                "use_cases": ["High-dimensional data", "Text classification", "Image recognition", "Gene classification"],
                "pros": ["Effective in high dimensions", "Memory efficient", "Versatile with kernels", "Works well with small datasets"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling", "No probability estimates", "Hard to interpret"],
                "parameters": {
                    "C": {
                        "description": "Regularization parameter controlling trade-off between smooth decision boundary and classifying training points correctly",
                        "range": "0.001 to 1000, typically 0.1 to 100",
                        "effect": "Lower C = wider margin, more regularization (underfitting). Higher C = narrower margin, less regularization (overfitting)",
                        "tuning_tips": "Start with 1.0, increase for complex data, decrease for simple patterns"
                    },
                    "kernel": {
                        "description": "Kernel function for transforming data to higher dimensions",
                        "options": {
                            "linear": "No transformation, for linearly separable data",
                            "rbf": "Radial Basis Function, good for non-linear data",
                            "poly": "Polynomial kernel for polynomial relationships",
                            "sigmoid": "Neural network-like kernel"
                        },
                        "effect": "Linear is fastest, RBF is most versatile, poly can overfit easily",
                        "recommendation": "Start with RBF, try linear if data seems linearly separable"
                    },
                    "gamma": {
                        "description": "Kernel coefficient defining influence of single training example",
                        "range": "0.001 to 1, typically 0.01 to 0.1",
                        "effect": "Lower gamma = far reach (underfitting), Higher gamma = close reach (overfitting)",
                        "tuning_tips": "Use 'scale' or 'auto' initially, then fine-tune manually"
                    }
                },
                "preprocessing_requirements": ["Feature scaling mandatory", "Handle outliers", "Consider feature selection"],
                "complexity": "O(n²) to O(n³) for training, O(n_sv * d) for prediction"
            },
            
            "Decision Tree": {
                "name": "Decision Tree",
                "description": "Tree-like model making decisions by splitting data based on feature values to maximize information gain or minimize impurity.",
                "mathematical_foundation": "Information Gain = H(parent) - Σ(|child|/|parent| * H(child)). Gini Impurity = 1 - Σpi²",
                "use_cases": ["Interpretable models", "Feature selection", "Mixed data types", "Rule extraction"],
                "pros": ["Highly interpretable", "No assumptions about data", "Handles mixed data types", "Automatic feature selection"],
                "cons": ["Prone to overfitting", "Unstable (small changes = different tree)", "Biased toward features with many levels", "Poor extrapolation"],
                "parameters": {
                    "max_depth": {
                        "description": "Maximum depth of the tree to control overfitting",
                        "range": "1 to 50, typically 3-20, None for unlimited",
                        "effect": "Deeper trees capture more patterns but overfit. Shallow trees underfit but generalize better",
                        "tuning_tips": "Start with 3-10, use cross-validation to find optimal depth"
                    },
                    "min_samples_split": {
                        "description": "Minimum samples required to split an internal node",
                        "range": "2 to 20, typically 2-10",
                        "effect": "Higher values prevent overfitting by requiring more samples for splits",
                        "recommendation": "Increase for noisy data, keep low for clean data"
                    },
                    "min_samples_leaf": {
                        "description": "Minimum samples required in a leaf node",
                        "range": "1 to 10, typically 1-5",
                        "effect": "Higher values smooth decision boundary and prevent overfitting",
                        "tuning_tips": "Increase if model overfits, especially with small datasets"
                    },
                    "criterion": {
                        "description": "Function to measure split quality",
                        "options": {
                            "gini": "Gini impurity, faster computation",
                            "entropy": "Information gain, may give slightly better results",
                            "squared_error": "MSE for regression"
                        },
                        "effect": "Gini and entropy usually give similar results, entropy is more computationally expensive",
                        "recommendation": "Start with Gini, try entropy if results are poor"
                    }
                },
                "preprocessing_requirements": ["Handle missing values", "Feature scaling not required", "Consider pruning"],
                "complexity": "O(n*log(n)*d) for training, O(log(n)) for prediction"
            },
            
            "Linear Model": {
                "name": "Linear Model",
                "description": "Linear relationships between features and target using least squares (regression) or maximum likelihood (classification).",
                "mathematical_foundation": "y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ. For classification: P(y=1|x) = 1/(1+e^(-w·x))",
                "use_cases": ["Baseline models", "Linear relationships", "Feature importance", "Fast predictions"],
                "pros": ["Fast training and prediction", "Interpretable coefficients", "No hyperparameters to tune", "Good baseline"],
                "cons": ["Assumes linear relationships", "Sensitive to outliers", "Poor with non-linear data", "Requires feature scaling"],
                "parameters": {
                    "fit_intercept": {
                        "description": "Whether to calculate intercept term",
                        "options": {"True": "Include bias term", "False": "Force line through origin"},
                        "effect": "Usually True unless you know the relationship passes through origin",
                        "recommendation": "Keep True in most cases"
                    },
                    "normalize": {
                        "description": "Whether to normalize features before fitting",
                        "options": {"True": "Normalize to unit norm", "False": "Use features as-is"},
                        "effect": "Can help with numerical stability",
                        "recommendation": "Use StandardScaler instead for better control"
                    }
                },
                "preprocessing_requirements": ["Feature scaling recommended", "Handle outliers", "Check linear assumptions"],
                "complexity": "O(n*d) for training and prediction"
            },
            
            "Random Forest": {
                "name": "Random Forest",
                "description": "Ensemble of decision trees using bootstrap sampling and random feature selection to reduce overfitting and improve generalization.",
                "mathematical_foundation": "Bagging: Prediction = (1/B)∑T_b(x) where T_b is trained on bootstrap sample b",
                "use_cases": ["Robust general-purpose model", "Feature importance", "Large datasets", "Mixed data types"],
                "pros": ["Reduces overfitting", "Handles missing values", "Provides feature importance", "Robust to outliers"],
                "cons": ["Less interpretable", "Can overfit with very noisy data", "Memory intensive", "May not perform well on sparse data"],
                "parameters": {
                    "n_estimators": {
                        "description": "Number of trees in the forest",
                        "range": "10 to 1000, typically 50-500",
                        "effect": "More trees = better performance but longer training time. Diminishing returns after certain point",
                        "tuning_tips": "Start with 100, increase until performance plateaus"
                    },
                    "max_features": {
                        "description": "Number of features to consider for best split",
                        "options": {
                            "sqrt": "Square root of total features (recommended for classification)",
                            "log2": "Log base 2 of total features",
                            "auto": "Same as sqrt",
                            "None": "Use all features"
                        },
                        "effect": "Fewer features increase randomness and reduce overfitting",
                        "recommendation": "sqrt for classification, 1/3 of features for regression"
                    },
                    "max_depth": {
                        "description": "Maximum depth of individual trees",
                        "range": "1 to 50, typically 10-20, None for unlimited",
                        "effect": "Deeper trees capture more patterns but may overfit despite ensemble averaging",
                        "tuning_tips": "Start with None, reduce if overfitting occurs"
                    },
                    "min_samples_split": {
                        "description": "Minimum samples required to split internal node",
                        "range": "2 to 20, typically 2-10",
                        "effect": "Higher values prevent overfitting in individual trees",
                        "recommendation": "Increase for noisy data or small datasets"
                    }
                },
                "preprocessing_requirements": ["Handle missing values", "Feature scaling not critical", "Consider feature selection for very high dimensions"],
                "complexity": "O(n*log(n)*d*B) for training where B is number of trees"
            },
            
            "XGBoost": {
                "name": "Extreme Gradient Boosting",
                "description": "Advanced gradient boosting framework that builds models sequentially, with each model correcting errors of previous models using optimized gradient descent.",
                "mathematical_foundation": "Objective = Σl(yi, ŷi) + Σ(γTj + ½λ||wj||²) where l is loss, γ,λ are regularization",
                "use_cases": ["Competitions", "Structured data", "Large datasets", "Feature importance analysis"],
                "pros": ["State-of-the-art performance", "Built-in regularization", "Handles missing values", "Parallel processing"],
                "cons": ["Many hyperparameters", "Requires tuning", "Can overfit easily", "Memory intensive"],
                "parameters": {
                    "n_estimators": {
                        "description": "Number of boosting rounds (trees to build)",
                        "range": "50 to 5000, typically 100-1000",
                        "effect": "More estimators = better training performance but higher overfitting risk",
                        "tuning_tips": "Use early stopping to find optimal number automatically"
                    },
                    "learning_rate": {
                        "description": "Step size shrinkage to prevent overfitting",
                        "range": "0.01 to 0.3, typically 0.05-0.15",
                        "effect": "Lower rate = more robust but needs more estimators. Higher rate = faster but may overfit",
                        "tuning_tips": "Lower learning rate with more estimators often works better"
                    },
                    "max_depth": {
                        "description": "Maximum depth of individual trees",
                        "range": "3 to 10, typically 3-8",
                        "effect": "Deeper trees model more complex interactions but risk overfitting",
                        "recommendation": "Start with 6, reduce if overfitting, increase for complex data"
                    },
                    "subsample": {
                        "description": "Fraction of samples used for each tree",
                        "range": "0.5 to 1.0, typically 0.8-1.0",
                        "effect": "Lower values prevent overfitting through stochastic training",
                        "tuning_tips": "0.8-0.9 often works well, lower for very large datasets"
                    },
                    "colsample_bytree": {
                        "description": "Fraction of features used for each tree",
                        "range": "0.3 to 1.0, typically 0.8-1.0",
                        "effect": "Lower values increase randomness and reduce overfitting",
                        "recommendation": "0.8-1.0 for most cases, lower for high-dimensional data"
                    }
                },
                "preprocessing_requirements": ["Handle missing values (XGBoost can handle some)", "Consider feature scaling", "Feature engineering important"],
                "complexity": "O(n*log(n)*d*B) for training, efficient parallel implementation"
            }
        }
    
    @staticmethod
    def get_preprocessing_docs() -> Dict[str, Dict[str, Any]]:
        """Returns documentation for preprocessing techniques."""
        return {
            "Missing Values": {
                "description": "Strategies for handling missing data points in features",
                "methods": {
                    "Mean/Median Imputation": "Replace missing values with mean (normal distribution) or median (skewed distribution)",
                    "Mode Imputation": "Replace missing categorical values with most frequent category",
                    "Forward/Backward Fill": "Use previous/next value for time series data",
                    "KNN Imputation": "Use k-nearest neighbors to estimate missing values",
                    "Iterative Imputation": "Model each feature with missing values as function of other features"
                },
                "considerations": [
                    "Missing at Random (MAR) vs Missing Not at Random (MNAR)",
                    "Amount of missing data (>20% may require dropping)",
                    "Pattern of missingness across features",
                    "Domain knowledge about why data is missing"
                ]
            },
            
            "Feature Scaling": {
                "description": "Normalizing feature ranges to ensure all features contribute equally",
                "methods": {
                    "StandardScaler": "Z-score normalization: (x - μ) / σ. Mean=0, Std=1",
                    "MinMaxScaler": "Scale to range [0,1]: (x - min) / (max - min)",
                    "RobustScaler": "Use median and IQR: (x - median) / IQR. Robust to outliers",
                    "Normalizer": "Scale samples individually to unit norm"
                },
                "when_to_use": {
                    "StandardScaler": "Normal distribution, few outliers, SVM/KNN/Neural Networks",
                    "MinMaxScaler": "Bounded range needed, preserve zero values",
                    "RobustScaler": "Many outliers present, skewed distributions"
                }
            },
            
            "Feature Engineering": {
                "description": "Creating new features from existing ones to improve model performance",
                "techniques": {
                    "Polynomial Features": "Create interactions and polynomial terms: x1*x2, x1², etc.",
                    "Binning": "Convert continuous variables to categorical bins",
                    "Log Transform": "Apply log(x) to reduce skewness in right-skewed data",
                    "Date Features": "Extract day, month, year, weekday from datetime columns",
                    "Aggregations": "Group-by statistics: mean, max, count by category"
                },
                "mathematical_foundations": {
                    "Box-Cox Transform": "y = (x^λ - 1) / λ for λ≠0, ln(x) for λ=0",
                    "Yeo-Johnson": "Extension of Box-Cox for negative values",
                    "Target Encoding": "Replace category with target mean for that category"
                }
            }
        }
    
    @staticmethod
    def get_evaluation_metrics_docs() -> Dict[str, Dict[str, Any]]:
        """Returns documentation for evaluation metrics."""
        return {
            "Classification Metrics": {
                "Accuracy": {
                    "formula": "(TP + TN) / (TP + TN + FP + FN)",
                    "interpretation": "Proportion of correct predictions",
                    "best_for": "Balanced datasets",
                    "limitations": "Misleading for imbalanced data"
                },
                "Precision": {
                    "formula": "TP / (TP + FP)",
                    "interpretation": "Of positive predictions, how many were correct?",
                    "best_for": "When false positives are costly",
                    "example": "Medical diagnosis - avoid false positive diagnoses"
                },
                "Recall (Sensitivity)": {
                    "formula": "TP / (TP + FN)",
                    "interpretation": "Of actual positives, how many were found?",
                    "best_for": "When false negatives are costly",
                    "example": "Disease screening - don't miss actual cases"
                },
                "F1-Score": {
                    "formula": "2 * (Precision * Recall) / (Precision + Recall)",
                    "interpretation": "Harmonic mean of precision and recall",
                    "best_for": "Imbalanced datasets, overall performance",
                    "range": "0 to 1, higher is better"
                },
                "ROC-AUC": {
                    "formula": "Area under ROC curve (TPR vs FPR)",
                    "interpretation": "Probability model ranks random positive higher than random negative",
                    "best_for": "Binary classification, probability predictions",
                    "range": "0.5 to 1.0, 0.5 = random, 1.0 = perfect"
                }
            },
            
            "Regression Metrics": {
                "RMSE": {
                    "formula": "√(Σ(yi - ŷi)² / n)",
                    "interpretation": "Average prediction error in original units",
                    "best_for": "When large errors are particularly bad",
                    "sensitivity": "Sensitive to outliers"
                },
                "MAE": {
                    "formula": "Σ|yi - ŷi| / n",
                    "interpretation": "Average absolute prediction error",
                    "best_for": "Robust to outliers, interpretable",
                    "comparison": "Less sensitive to outliers than RMSE"
                },
                "R²": {
                    "formula": "1 - SS_res/SS_tot = 1 - Σ(yi-ŷi)²/Σ(yi-ȳ)²",
                    "interpretation": "Proportion of variance explained by model",
                    "range": "-∞ to 1, negative means worse than mean prediction",
                    "limitations": "Can be misleading with non-linear relationships"
                },
                "MAPE": {
                    "formula": "Σ|yi - ŷi|/|yi| / n * 100%",
                    "interpretation": "Average percentage error",
                    "best_for": "When relative errors matter more than absolute",
                    "limitations": "Undefined when yi = 0, biased toward low values"
                }
            }
        }
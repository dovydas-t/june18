from typing import Dict, List, Any, Optional

class MenuManager:
    """Centralized menu system manager."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.ui = pipeline.ui
    
    def handle_data_loading_menu(self):
        """Enhanced data loading menu with clean previous choices option."""
        while True:
            self.ui.print("\n📁 DATA LOADING & CONFIGURATION v3.1")
            self.ui.print("="*50)
            
            # Option to clear previous choices
            self.ui.print("🔧 Options:")
            self.ui.print("C. Clear previous dataset choices")
            
            # Show existing datasets from database
            existing_datasets = self.pipeline.data_ops.get_existing_datasets()
            
            if existing_datasets:
                self.ui.print("\n📋 Available Datasets:")
                for i, dataset in enumerate(existing_datasets, 1):
                    self.ui.print(f"{i:2d}. {dataset['name']} ({dataset['problem_type']}, {dataset['n_samples']} samples)")
                
                self.ui.print(f"{len(existing_datasets)+1:2d}. Load new dataset from file")
                self.ui.print(f" 0. Go Back")
                
                choice = self.ui.input(f"Enter choice (C/1-{len(existing_datasets)+1}/0)", default="0")
                
                if choice.upper() == "C":
                    self.pipeline.data_ops.clear_dataset_cache()
                    continue
                
                try:
                    choice = int(choice)
                    if choice == 0:
                        return
                    elif 1 <= choice <= len(existing_datasets):
                        self._load_existing_dataset(existing_datasets[choice-1])
                        break
                    elif choice == len(existing_datasets) + 1:
                        if self._load_new_dataset():
                            break
                    else:
                        self.ui.print("❌ Invalid choice.")
                        input("Press Enter to continue...")
                except ValueError:
                    self.ui.print("❌ Please enter a valid number or 'C'.")
                    input("Press Enter to continue...")
            else:
                if self._load_new_dataset():
                    break
                else:
                    return
    
    def _show_visualization_gallery(self):
        """Enhanced visualization gallery with smart recommendations."""
        if self.pipeline.train_data is None:
            self.ui.print("❌ No data loaded.")
            return
        
        self.ui.print("\n🎨 SMART VISUALIZATION GALLERY v3.1")
        self.ui.print("="*45)
        
        df = self.pipeline.train_data
        target_col = self.pipeline.target_column
        
        # Analyze data and provide recommendations
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        self.ui.print("📊 Available Visualizations:")
        
        # Distribution analysis
        if numerical_cols:
            self.ui.print("\n1. 📈 Distribution Analysis")
            self.ui.print("   • Histograms with normality tests")
            self.ui.print("   • Box plots for outlier detection")
            self.ui.print("   • Q-Q plots for distribution assessment")
            
        # Correlation analysis
        if len(numerical_cols) > 1:
            self.ui.print("\n2. 🔗 Correlation Heatmap")
            self.ui.print("   • Feature correlation matrix")
            self.ui.print("   • Multicollinearity detection")
            
        # Target analysis
        if target_col:
            self.ui.print("\n3. 🎯 Target Variable Analysis")
            if self.pipeline.problem_type == 'classification':
                self.ui.print("   • Class distribution")
                self.ui.print("   • Feature vs target relationships")
            else:
                self.ui.print("   • Target distribution")
                self.ui.print("   • Feature-target scatter plots")
        
        # Missing values pattern
        if df.isnull().sum().sum() > 0:
            self.ui.print("\n4. 📋 Missing Values Patterns")
            self.ui.print("   • Missing data heatmap")
            self.ui.print("   • Missing patterns analysis")
        
        self.ui.print("\n💡 Smart Recommendations:")
        self._provide_visualization_recommendations(df, target_col)
    
    def _provide_visualization_recommendations(self, df, target_col):
        """Provide smart visualization recommendations."""
        recommendations = []
        
        # Check skewed distributions
        numerical_cols = df.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if col != target_col:
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    recommendations.append(f"📈 {col} is skewed (skew={skewness:.2f}) - consider log transformation")
        
        # Check missing data
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            recommendations.append(f"📋 {len(high_missing)} columns have >20% missing data - review imputation strategy")
        
        # Check categorical cardinality
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col:
                cardinality = df[col].nunique()
                if cardinality > 50:
                    recommendations.append(f"🏷️ {col} has high cardinality ({cardinality}) - consider grouping rare categories")
        
        for i, rec in enumerate(recommendations, 1):
            self.ui.print(f"  {i}. {rec}")
        
        if not recommendations:
            self.ui.print("  ✅ Data looks good for standard preprocessing!")
    
    def _interactive_missing_values(self):
        """Interactive missing values handling."""
        if self.pipeline.train_data is None:
            self.ui.print("❌ No data loaded.")
            return
        
        self.ui.print("\n📋 INTERACTIVE MISSING VALUES HANDLER v3.1")
        self.ui.print("="*50)
        
        df = self.pipeline.train_data
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) == 0:
            self.ui.print("✅ No missing values detected!")
            return
        
        self.ui.print(f"Found missing values in {len(missing_cols)} columns:")
        
        # Initialize missing config if not exists
        if not hasattr(self.pipeline, 'missing_config'):
            self.pipeline.missing_config = {}
        
        for col in missing_cols.index:
            missing_count = missing_cols[col]
            missing_pct = (missing_count / len(df)) * 100
            
            self.ui.print(f"\n🔍 Column: {col}")
            self.ui.print(f"   Missing: {missing_count} ({missing_pct:.1f}%)")
            self.ui.print(f"   Type: {df[col].dtype}")
            
            # Provide smart recommendations
            recommendation = self._get_missing_value_recommendation(df, col, missing_pct)
            self.ui.print(f"   💡 Recommendation: {recommendation}")
            
            # Strategy options
            if df[col].dtype in ['int64', 'float64']:
                options = ["mean", "median", "constant", "drop"]
                self.ui.print("   Options: 1=Mean, 2=Median, 3=Constant, 4=Drop column")
            else:
                options = ["most_frequent", "constant", "drop"]
                self.ui.print("   Options: 1=Most frequent, 2=Constant, 3=Drop column")
            
            choice = self.ui.input(f"   Strategy for {col} (1-{len(options)})", default="1")
            
            try:
                strategy_idx = int(choice) - 1
                if 0 <= strategy_idx < len(options):
                    self.pipeline.missing_config[col] = options[strategy_idx]
                    self.ui.print(f"   ✅ Set strategy: {options[strategy_idx]}")
            except ValueError:
                self.ui.print("   ⚠️ Invalid choice, using default")
        
        self.ui.print(f"\n✅ Missing values configuration completed!")
    
    def _get_missing_value_recommendation(self, df, col, missing_pct):
        """Get smart recommendation for missing value handling."""
        if missing_pct > 50:
            return "Drop column (too much missing data)"
        elif missing_pct > 20:
            return "Consider advanced imputation or feature engineering"
        elif df[col].dtype in ['int64', 'float64']:
            skewness = abs(df[col].skew())
            if skewness > 1:
                return "Use median (data is skewed)"
            else:
                return "Use mean (data is approximately normal)"
        else:
            return "Use most frequent value"
    
    def _interactive_feature_scaling(self):
        """Interactive feature scaling configuration."""
        self.ui.print("\n📏 INTERACTIVE FEATURE SCALING v3.1")
        self.ui.print("="*45)
        
        if self.pipeline.train_data is None:
            self.ui.print("❌ No data loaded.")
            return
        
        numerical_cols = self.pipeline.train_data.select_dtypes(include=['number']).columns
        if len(numerical_cols) == 0:
            self.ui.print("✅ No numerical features to scale.")
            return
        
        self.ui.print(f"Found {len(numerical_cols)} numerical features to scale.")
        
        # Analyze data characteristics
        self._analyze_scaling_requirements(self.pipeline.train_data, numerical_cols)
        
        self.ui.print("\n🔧 Scaling Methods:")
        self.ui.print("1. StandardScaler - Mean=0, Std=1 (best for normal distributions)")
        self.ui.print("2. MinMaxScaler - Scale to [0,1] range (preserves relationships)")
        self.ui.print("3. RobustScaler - Uses median/IQR (robust to outliers)")
        self.ui.print("4. No scaling - Keep original values")
        
        choice = self.ui.input("Select scaling method (1-4)", default="1")
        
        scaling_methods = {
            "1": "standard",
            "2": "minmax", 
            "3": "robust",
            "4": "none"
        }
        
        method = scaling_methods.get(choice, "standard")
        self.pipeline.scaling_method = method
        
        self.ui.print(f"✅ Feature scaling method set to: {method}")
    
    def _analyze_scaling_requirements(self, df, numerical_cols):
        """Analyze data to recommend best scaling method."""
        self.ui.print("\n📊 Data Analysis for Scaling:")
        
        has_outliers = False
        has_different_scales = False
        
        ranges = []
        for col in numerical_cols:
            col_range = df[col].max() - df[col].min()
            ranges.append(col_range)
            
            # Check for outliers using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            
            if outliers > len(df) * 0.05:  # More than 5% outliers
                has_outliers = True
                self.ui.print(f"   ⚠️ {col}: {outliers} outliers detected")
        
        # Check scale differences
        if len(ranges) > 1:
            max_range = max(ranges)
            min_range = min(ranges)
            if max_range / min_range > 100:
                has_different_scales = True
                self.ui.print(f"   📏 Large scale differences detected (ratio: {max_range/min_range:.1f})")
        
        # Provide recommendation
        self.ui.print("\n💡 Scaling Recommendation:")
        if has_outliers:
            self.ui.print("   🎯 RobustScaler recommended (many outliers present)")
        elif has_different_scales:
            self.ui.print("   🎯 StandardScaler or MinMaxScaler recommended (different scales)")
        else:
            self.ui.print("   🎯 StandardScaler recommended (general purpose)")
    
    def _manual_preprocessing_menu(self):
        """Enhanced manual preprocessing configuration menu."""
        while True:
            self.ui.print("\n🔧 MANUAL PREPROCESSING CONFIGURATION v3.1")
            self.ui.print("="*55)
            
            if self.pipeline.train_data is None:
                self.ui.print("❌ No data loaded.")
                input("Press Enter to continue...")
                return
            
            # Show current configuration
            self._show_current_preprocessing_config()
            
            self.ui.print("\n🛠️ Configuration Options:")
            self.ui.print("1. 📋 Configure missing values handling")
            self.ui.print("2. 📏 Configure feature scaling")
            self.ui.print("3. 🎯 Configure outlier removal")
            self.ui.print("4. 🏷️ Review categorical encoding")
            self.ui.print("5. 🔧 Data transformation tools")
            self.ui.print("6. 📊 Create new features")
            self.ui.print("7. 🗑️ Drop selected columns")
            self.ui.print("8. ✅ Apply all configurations")
            self.ui.print("0. 🔙 Back to main menu")
            
            choice = self.ui.input("Enter choice (0-8)", default="0")
            
            if choice == "0":
                break
            elif choice == "1":
                self._interactive_missing_values()
            elif choice == "2":
                self._interactive_feature_scaling()
            elif choice == "3":
                self._configure_outlier_removal()
            elif choice == "4":
                self._review_categorical_encoding()
            elif choice == "5":
                self._data_transformation_tools()
            elif choice == "6":
                self._create_new_features()
            elif choice == "7":
                self._drop_selected_columns()
            elif choice == "8":
                success = self.pipeline.preprocessor.preprocess_data(self.pipeline)
                if success:
                    self.ui.print("✅ All preprocessing configurations applied!")
                input("Press Enter to continue...")
                break
            else:
                self.ui.print("❌ Invalid choice.")
            
            if choice != "0" and choice != "8":
                input("Press Enter to continue...")
    
    def _show_current_preprocessing_config(self):
        """Show current preprocessing configuration."""
        self.ui.print("📋 Current Configuration:")
        
        # Missing values
        missing_config = getattr(self.pipeline, 'missing_config', {})
        if missing_config:
            self.ui.print(f"   📋 Missing values: {len(missing_config)} columns configured")
        else:
            self.ui.print("   📋 Missing values: Auto-detection enabled")
        
        # Scaling
        scaling_method = getattr(self.pipeline, 'scaling_method', 'standard')
        self.ui.print(f"   📏 Feature scaling: {scaling_method}")
        
        # Outliers
        outlier_method = getattr(self.pipeline, 'outlier_method', None)
        if outlier_method:
            threshold = getattr(self.pipeline, 'outlier_threshold', 1.5)
            self.ui.print(f"   🎯 Outlier removal: {outlier_method} (threshold: {threshold})")
        else:
            self.ui.print("   🎯 Outlier removal: Disabled")
    
    def _configure_outlier_removal(self):
        """Configure outlier removal settings."""
        self.ui.print("\n🎯 OUTLIER REMOVAL CONFIGURATION")
        self.ui.print("="*40)
        
        self.ui.print("Methods:")
        self.ui.print("1. IQR method (Interquartile Range)")
        self.ui.print("2. Z-score method")
        self.ui.print("3. Disable outlier removal")
        
        choice = self.ui.input("Select method (1-3)", default="3")
        
        if choice == "1":
            self.pipeline.outlier_method = "iqr"
            threshold = float(self.ui.input("IQR threshold (1.5=standard, 3.0=conservative)", default="1.5"))
            self.pipeline.outlier_threshold = threshold
        elif choice == "2":
            self.pipeline.outlier_method = "zscore"
            threshold = float(self.ui.input("Z-score threshold (3.0=standard)", default="3.0"))
            self.pipeline.outlier_threshold = threshold
        else:
            self.pipeline.outlier_method = None
        
        self.ui.print(f"✅ Outlier removal configured")
    
    def _data_transformation_tools(self):
        """Advanced data transformation tools."""
        self.ui.print("\n🔧 DATA TRANSFORMATION TOOLS v3.1")
        self.ui.print("="*45)
        
        numerical_cols = self.pipeline.train_data.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) == 0:
            self.ui.print("❌ No numerical columns for transformation.")
            return
        
        self.ui.print("Available transformations:")
        self.ui.print("1. Log transformation (for right-skewed data)")
        self.ui.print("2. Square root transformation")
        self.ui.print("3. Box-Cox transformation")
        self.ui.print("4. Binning (convert continuous to categorical)")
        
        # Analyze and recommend transformations
        self._analyze_transformation_needs(numerical_cols)
    
    def _analyze_transformation_needs(self, numerical_cols):
        """Analyze which columns need transformation."""
        self.ui.print("\n📊 Transformation Analysis:")
        
        df = self.pipeline.train_data
        recommendations = []
        
        for col in numerical_cols:
            if col == self.pipeline.target_column:
                continue
            
            skewness = df[col].skew()
            
            if skewness > 1:
                recommendations.append(f"📈 {col}: Right-skewed (skew={skewness:.2f}) → Log transformation")
            elif skewness < -1:
                recommendations.append(f"📈 {col}: Left-skewed (skew={skewness:.2f}) → Power transformation")
            
            # Check for zero/negative values
            if (df[col] <= 0).any() and skewness > 1:
                recommendations.append(f"⚠️ {col}: Contains zero/negative values → Add constant before log")
        
        if recommendations:
            self.ui.print("💡 Recommendations:")
            for rec in recommendations:
                self.ui.print(f"   {rec}")
        else:
            self.ui.print("✅ No obvious transformations needed")
    
    def _create_new_features(self):
        """Interactive feature creation tool."""
        self.ui.print("\n📊 FEATURE CREATION TOOL v3.1")
        self.ui.print("="*40)
        
        df = self.pipeline.train_data
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if self.pipeline.target_column in numerical_cols:
            numerical_cols.remove(self.pipeline.target_column)
        
        if len(numerical_cols) < 2:
            self.ui.print("❌ Need at least 2 numerical columns for feature creation.")
            return
        
        self.ui.print("Feature creation options:")
        self.ui.print("1. Sum selected columns")
        self.ui.print("2. Product of selected columns")
        self.ui.print("3. Ratio of two columns")
        self.ui.print("4. Polynomial features")
        
        choice = self.ui.input("Select option (1-4)", default="1")
        
        if choice == "1":
            self._sum_columns_feature(numerical_cols)
        elif choice == "2":
            self._product_columns_feature(numerical_cols)
        elif choice == "3":
            self._ratio_columns_feature(numerical_cols)
        elif choice == "4":
            self._polynomial_features(numerical_cols)
    
    def _sum_columns_feature(self, numerical_cols):
        """Create sum feature from selected columns."""
        self.ui.print("\nAvailable columns:")
        for i, col in enumerate(numerical_cols, 1):
            self.ui.print(f"{i}. {col}")
        
        selected = self.ui.input("Enter column numbers to sum (comma-separated)", default="1,2")
        
        try:
            indices = [int(x.strip()) - 1 for x in selected.split(',')]
            selected_cols = [numerical_cols[i] for i in indices if 0 <= i < len(numerical_cols)]
            
            if len(selected_cols) >= 2:
                new_feature_name = self.ui.input("Name for new feature", default="sum_feature")
                
                # Create the new feature
                self.pipeline.train_data[new_feature_name] = self.pipeline.train_data[selected_cols].sum(axis=1)
                
                if self.pipeline.test_data is not None:
                    # Check if columns exist in test data
                    available_cols = [col for col in selected_cols if col in self.pipeline.test_data.columns]
                    if available_cols:
                        self.pipeline.test_data[new_feature_name] = self.pipeline.test_data[available_cols].sum(axis=1)
                
                self.ui.print(f"✅ Created feature '{new_feature_name}' from {len(selected_cols)} columns")
            else:
                self.ui.print("❌ Need at least 2 columns")
        except (ValueError, IndexError):
            self.ui.print("❌ Invalid column selection")
    
    def _drop_selected_columns(self):
        """Interactive column dropping tool."""
        self.ui.print("\n🗑️ COLUMN DROPPING TOOL v3.1")
        self.ui.print("="*35)
        
        df = self.pipeline.train_data
        all_cols = [col for col in df.columns if col != self.pipeline.target_column]
        
        self.ui.print("Available columns to drop:")
        for i, col in enumerate(all_cols, 1):
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_pct = (df[col].nunique() / len(df)) * 100
            
            # Provide recommendations
            recommendation = ""
            if missing_pct > 50:
                recommendation = " [RECOMMEND DROP: >50% missing]"
            elif unique_pct > 95:
                recommendation = " [RECOMMEND DROP: likely ID column]"
            elif df[col].nunique() == 1:
                recommendation = " [RECOMMEND DROP: constant column]"
            
            self.ui.print(f"{i:2d}. {col} (missing: {missing_pct:.1f}%, unique: {unique_pct:.1f}%){recommendation}")
        
        selected = self.ui.input("Enter column numbers to drop (comma-separated)", default="")
        
        if selected.strip():
            try:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                cols_to_drop = [all_cols[i] for i in indices if 0 <= i < len(all_cols)]
                
                if cols_to_drop:
                    confirm = self.ui.confirm(f"Drop {len(cols_to_drop)} columns: {cols_to_drop}?", default=False)
                    
                    if confirm:
                        self.pipeline.train_data = self.pipeline.train_data.drop(columns=cols_to_drop)
                        
                        if self.pipeline.test_data is not None:
                            available_drops = [col for col in cols_to_drop if col in self.pipeline.test_data.columns]
                            if available_drops:
                                self.pipeline.test_data = self.pipeline.test_data.drop(columns=available_drops)
                        
                        self.ui.print(f"✅ Dropped {len(cols_to_drop)} columns")
                    else:
                        self.ui.print("❌ Operation cancelled")
            except (ValueError, IndexError):
                self.ui.print("❌ Invalid column selection")
    
    def _product_columns_feature(self, numerical_cols):
        """Create product feature from selected columns."""
        self.ui.print("\nAvailable columns:")
        for i, col in enumerate(numerical_cols, 1):
            self.ui.print(f"{i}. {col}")
        
        selected = self.ui.input("Enter column numbers for product (comma-separated)", default="1,2")
        
        try:
            indices = [int(x.strip()) - 1 for x in selected.split(',')]
            selected_cols = [numerical_cols[i] for i in indices if 0 <= i < len(numerical_cols)]
            
            if len(selected_cols) >= 2:
                new_feature_name = self.ui.input("Name for new feature", default="product_feature")
                
                # Create the new feature
                self.pipeline.train_data[new_feature_name] = self.pipeline.train_data[selected_cols].prod(axis=1)
                
                if self.pipeline.test_data is not None:
                    available_cols = [col for col in selected_cols if col in self.pipeline.test_data.columns]
                    if available_cols:
                        self.pipeline.test_data[new_feature_name] = self.pipeline.test_data[available_cols].prod(axis=1)
                
                self.ui.print(f"✅ Created feature '{new_feature_name}' from {len(selected_cols)} columns")
            else:
                self.ui.print("❌ Need at least 2 columns")
        except (ValueError, IndexError):
            self.ui.print("❌ Invalid column selection")
    
    def _ratio_columns_feature(self, numerical_cols):
        """Create ratio feature from two columns."""
        self.ui.print("\nAvailable columns:")
        for i, col in enumerate(numerical_cols, 1):
            self.ui.print(f"{i}. {col}")
        
        numerator_idx = int(self.ui.input("Select numerator column number", default="1")) - 1
        denominator_idx = int(self.ui.input("Select denominator column number", default="2")) - 1
        
        try:
            if 0 <= numerator_idx < len(numerical_cols) and 0 <= denominator_idx < len(numerical_cols):
                num_col = numerical_cols[numerator_idx]
                den_col = numerical_cols[denominator_idx]
                
                new_feature_name = self.ui.input("Name for new feature", default=f"{num_col}_over_{den_col}")
                
                # Create ratio with zero division protection
                self.pipeline.train_data[new_feature_name] = (
                    self.pipeline.train_data[num_col] / 
                    (self.pipeline.train_data[den_col] + 1e-10)
                )
                
                if self.pipeline.test_data is not None and num_col in self.pipeline.test_data.columns and den_col in self.pipeline.test_data.columns:
                    self.pipeline.test_data[new_feature_name] = (
                        self.pipeline.test_data[num_col] / 
                        (self.pipeline.test_data[den_col] + 1e-10)
                    )
                
                self.ui.print(f"✅ Created ratio feature '{new_feature_name}'")
            else:
                self.ui.print("❌ Invalid column selection")
        except (ValueError, IndexError, ZeroDivisionError):
            self.ui.print("❌ Error creating ratio feature")
    
    def _polynomial_features(self, numerical_cols):
        """Create polynomial features."""
        self.ui.print("\n📊 Polynomial Feature Creation")
        self.ui.print("⚠️ Warning: This can create many features quickly!")
        
        max_cols = min(5, len(numerical_cols))
        selected_cols = numerical_cols[:max_cols]
        
        self.ui.print(f"Using first {len(selected_cols)} columns: {selected_cols}")
        
        degree = int(self.ui.input("Polynomial degree (2-3 recommended)", default="2"))
        
        if degree > 3:
            self.ui.print("⚠️ High degree may create too many features")
            if not self.ui.confirm("Continue anyway?", default=False):
                return
        
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            
            # Create polynomial features
            train_poly = poly.fit_transform(self.pipeline.train_data[selected_cols])
            feature_names = poly.get_feature_names_out(selected_cols)
            
            # Add new features to dataframe
            for i, name in enumerate(feature_names):
                if name not in selected_cols:  # Skip original features
                    self.pipeline.train_data[f"poly_{name}"] = train_poly[:, i]
            
            # Apply to test data if available
            if self.pipeline.test_data is not None:
                available_cols = [col for col in selected_cols if col in self.pipeline.test_data.columns]
                if len(available_cols) == len(selected_cols):
                    test_poly = poly.transform(self.pipeline.test_data[selected_cols])
                    for i, name in enumerate(feature_names):
                        if name not in selected_cols:
                            self.pipeline.test_data[f"poly_{name}"] = test_poly[:, i]
            
            new_features = len(feature_names) - len(selected_cols)
            self.ui.print(f"✅ Created {new_features} polynomial features")
            
        except ImportError:
            self.ui.print("❌ scikit-learn not available for polynomial features")
        except Exception as e:
            self.ui.print(f"❌ Error creating polynomial features: {e}")
    
    def _review_categorical_encoding(self):
        """Review and configure categorical encoding."""
        self.ui.print("\n🏷️ CATEGORICAL ENCODING REVIEW")
        self.ui.print("="*40)
        
        df = self.pipeline.train_data
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if self.pipeline.target_column in categorical_cols:
            categorical_cols.remove(self.pipeline.target_column)
        
        if len(categorical_cols) == 0:
            self.ui.print("✅ No categorical columns found")
            return
        
        self.ui.print(f"Found {len(categorical_cols)} categorical columns:")
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().head(3).tolist()
            
            self.ui.print(f"\n📋 {col}:")
            self.ui.print(f"   Unique values: {unique_count}")
            self.ui.print(f"   Sample: {sample_values}")
            
            # Provide encoding recommendations
            if unique_count == 2:
                recommendation = "Binary encoding or Label encoding"
            elif unique_count <= 10:
                recommendation = "One-hot encoding or Label encoding"
            elif unique_count <= 50:
                recommendation = "Label encoding or Target encoding"
            else:
                recommendation = "Consider grouping rare categories first"
            
            self.ui.print(f"   💡 Recommendation: {recommendation}")
        
        self.ui.print("\n📝 Current strategy: Label encoding (automatic)")
        self.ui.print("   This converts categories to numerical labels")
    
    def show_main_menu_options(self):
        """Display main menu options consistently."""
        self.ui.print("\n" + "="*60)
        self.ui.print("🏠 ENHANCED ML PIPELINE v3.0 - MAIN MENU")
        self.ui.print("="*60)
        
        # Show current status
        if self.pipeline.train_data is not None:
            self.ui.print(f"📊 Dataset: {getattr(self.pipeline, 'dataset_name', 'Loaded')} "
                        f"({len(self.pipeline.train_data)} samples, {len(self.pipeline.train_data.columns)} columns)")
            if self.pipeline.X_train is not None:
                self.ui.print(f"🔧 Preprocessed: Ready for training")
            if self.pipeline.models:
                self.ui.print(f"🤖 Trained Models: {len(self.pipeline.models)} available")
        
        self.ui.print(f"\n📋 AVAILABLE OPTIONS:")
        self.ui.print("1. 📁 Load Data & Configure Dataset")
        self.ui.print("2. 🔍 Explore Data & Get Insights") 
        self.ui.print("3. 🔧 Configure Preprocessing Pipeline")
        self.ui.print("4. 🤖 Model Management & Documentation")
        self.ui.print("5. 🚀 Run Complete ML Pipeline")
        self.ui.print("6. 📊 View Results & Comparisons")
        self.ui.print("7. 💾 Generate Submission File")
        self.ui.print("8. 📚 Help & Documentation Center")
        self.ui.print("9. 🗄️ View Experiment Database")
        self.ui.print("10. ℹ️ About & Version Information")
        self.ui.print("11. ❌ Exit Pipeline")

    def _show_comprehensive_data_report(self):
        """Generate comprehensive data report."""
        self.ui.print("\n📋 COMPREHENSIVE DATA REPORT v3.1")
        self.ui.print("="*50)
        
        df = self.pipeline.train_data
        target_col = self.pipeline.target_column
        
        # Dataset Overview
        self.ui.print("📊 Dataset Overview:")
        self.ui.print(f"   Shape: {df.shape}")
        self.ui.print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        self.ui.print(f"   Target: {target_col} ({self.pipeline.problem_type})")
        
        # Data Quality Summary
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / df.size) * 100
        duplicates = df.duplicated().sum()
        
        self.ui.print(f"\n🔍 Data Quality:")
        self.ui.print(f"   Missing values: {missing_total:,} ({missing_pct:.1f}%)")
        self.ui.print(f"   Duplicate rows: {duplicates:,}")
        
        # Feature Summary
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        self.ui.print(f"\n📋 Features:")
        self.ui.print(f"   Numerical: {len(numerical_cols)}")
        self.ui.print(f"   Categorical: {len(categorical_cols)}")
        
        # Generate smart insights
        self.pipeline.exploration_engine.show_smart_data_insights(df, target_col)
    
    def _show_ai_insights_recommendations(self):
        """Show AI-powered insights and recommendations."""
        if self.pipeline.train_data is None:
            self.ui.print("❌ No data loaded.")
            return
        
        self.pipeline.exploration_engine.show_smart_data_insights(
            self.pipeline.train_data, 
            self.pipeline.target_column
        )
    
    def _show_feature_engineering_suggestions(self):
        """Show feature engineering suggestions."""
        if self.pipeline.train_data is None:
            self.ui.print("❌ No data loaded.")
            return
        
        self.ui.print("\n🔧 FEATURE ENGINEERING SUGGESTIONS v3.1")
        self.ui.print("="*50)
        
        self.ui.print("Available tools:")
        self.ui.print("1. 📊 Distribution fixer")
        self.ui.print("2. 🔧 Feature creation wizard")
        self.ui.print("3. 🏷️ Categorical optimization")
        self.ui.print("4. 📈 Smart transformations")
        
        choice = self.ui.input("Select tool (1-4)", default="1")
        
        if choice == "1":
            self.pipeline.exploration_engine.show_distribution_fixer(
                self.pipeline.train_data, 
                self.pipeline.target_column
            )
        elif choice == "2":
            self._create_new_features()
        elif choice == "3":
            self._review_categorical_encoding()
        elif choice == "4":
            self._data_transformation_tools()
    
    def _show_target_deep_dive(self):
        """Deep dive analysis of target variable."""
        if not self.pipeline.target_column:
            self.ui.print("❌ No target column specified.")
            return
        
        self.ui.print("\n🎯 TARGET VARIABLE DEEP DIVE v3.1")
        self.ui.print("="*45)
        
        target = self.pipeline.train_data[self.pipeline.target_column]
        
        self.ui.print(f"Target Column: {self.pipeline.target_column}")
        self.ui.print(f"Problem Type: {self.pipeline.problem_type}")
        self.ui.print(f"Missing Values: {target.isnull().sum()}")
        
        if self.pipeline.problem_type == 'classification':
            # Classification analysis
            value_counts = target.value_counts()
            self.ui.print(f"Number of Classes: {target.nunique()}")
            
            self.ui.print(f"\nClass Distribution:")
            for cls, count in value_counts.items():
                pct = (count / len(target)) * 100
                bar = "█" * int(pct / 2)  # Simple text bar
                self.ui.print(f"   {cls}: {count:,} ({pct:.1f}%) {bar}")
            
            # Balance analysis
            min_class_pct = (value_counts.min() / len(target)) * 100
            if min_class_pct < 5:
                self.ui.print(f"\n⚠️ Imbalanced classes detected (minimum class: {min_class_pct:.1f}%)")
                self.ui.print("💡 Consider: SMOTE, class weighting, or stratified sampling")
            else:
                self.ui.print(f"\n✅ Well-balanced classes")
        
        else:
            # Regression analysis
            stats = target.describe()
            skewness = target.skew()
            kurtosis = target.kurtosis()
            
            self.ui.print(f"\nTarget Statistics:")
            self.ui.print(f"   Range: {stats['min']:.3f} to {stats['max']:.3f}")
            self.ui.print(f"   Mean: {stats['mean']:.3f}")
            self.ui.print(f"   Median: {stats['50%']:.3f}")
            self.ui.print(f"   Std: {stats['std']:.3f}")
            self.ui.print(f"   Skewness: {skewness:.3f}")
            self.ui.print(f"   Kurtosis: {kurtosis:.3f}")
            
            # Distribution analysis
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                self.ui.print(f"\n📊 Target is {direction}-skewed")
                self.ui.print("💡 Consider: Log transformation or Box-Cox transformation")
            else:
                self.ui.print(f"\n✅ Target distribution is approximately normal")
        
        # Correlation with features (for numerical target)
        if self.pipeline.problem_type == 'regression':
            numerical_cols = self.pipeline.train_data.select_dtypes(include=['number']).columns
            numerical_cols = [col for col in numerical_cols if col != self.pipeline.target_column]
            
            if len(numerical_cols) > 0:
                correlations = self.pipeline.train_data[numerical_cols].corrwith(target).abs().sort_values(ascending=False)
                
                self.ui.print(f"\n🔗 Top Correlations with Target:")
                for col, corr in correlations.head(5).items():
                    strength = "Strong" if corr > 0.7 else "Moderate" if corr > 0.3 else "Weak"
                    self.ui.print(f"   {col}: {corr:.3f} ({strength})")


# Show about menu update for v3.1
    def _show_about_menu(self):
        """Show about information for v3.1."""
        self.ui.print("\n" + "="*60)
        self.ui.print("ℹ️ ENHANCED ML PIPELINE v3.1 - ABOUT")
        self.ui.print("="*60)
        
        self.ui.print("🚀 ENHANCED MACHINE LEARNING PIPELINE v3.1")
        self.ui.print("📅 Release Date: 2024")
        self.ui.print("👥 Authors: ML Pipeline Framework Team")
        self.ui.print("📄 License: MIT License")
        
        self.ui.print(f"\n🔥 NEW IN VERSION 3.1:")
        self.ui.print("• 🧠 Smart data insights and AI-powered recommendations")
        self.ui.print("• 📊 Interactive distribution fixer with transformation tools")
        self.ui.print("• 🔧 Enhanced feature creation wizard (sum, product, ratio, polynomial)")
        self.ui.print("• 🗑️ Smart column dropping with recommendations")
        self.ui.print("• 📋 Fixed dataset loading from folder paths")
        self.ui.print("• 🎯 Deep target variable analysis with balance detection")
        self.ui.print("• 🔍 Enhanced data exploration with visualization recommendations")
        self.ui.print("• 💡 Step-by-step data fixing guidance")
        
        self.ui.print(f"\n✅ FIXES IN VERSION 3.1:")
        self.ui.print("• 🔧 Fixed DataSplitter missing methods")
        self.ui.print("• 📋 Fixed MenuManager missing visualization and preprocessing methods")
        self.ui.print("• 🗂️ Improved dataset cache management")
        self.ui.print("• 🔄 Better error handling in feature creation")
        
        input("\nPress Enter to continue...")

    def handle_exploration_menu(self):
        """Handle data exploration menu."""
        while True:
            if self.pipeline.train_data is None:
                self.ui.print("❌ No data loaded. Please load data first.")
                input("Press Enter to continue...")
                return
            
            self._show_data_exploration_menu()
            
            choice = self.ui.input("Enter choice (0-9)", default="0")
            
            if choice == "0":
                return
            elif choice == "1":
                self.pipeline.exploration_engine.show_interactive_dashboard(
                    self.pipeline.train_data, self.pipeline.target_column, self.pipeline.problem_type)
            elif choice == "2":
                self.pipeline.exploration_engine.show_statistical_analysis_suite(
                    self.pipeline.train_data, self.pipeline.target_column)
            elif choice == "3":
                self._show_visualization_gallery()
            elif choice == "4":
                self.pipeline.exploration_engine.show_data_quality_assessment(
                    self.pipeline.train_data, self.pipeline.dataset_analysis)
            elif choice == "5":
                self._show_target_deep_dive()
            elif choice == "6":
                self.pipeline.exploration_engine.show_feature_relationships_analysis(
                    self.pipeline.train_data, self.pipeline.target_column)
            elif choice == "7":
                self._show_comprehensive_data_report()
            elif choice == "8":
                self._show_ai_insights_recommendations()
            elif choice == "9":
                self._show_feature_engineering_suggestions()
            else:
                self.ui.print("❌ Invalid choice.")
            
            input("\nPress Enter to continue...")

    def handle_preprocessing_menu(self):
        """Handle preprocessing configuration menu."""
        while True:
            if self.pipeline.train_data is None:
                self.ui.print("❌ No data loaded. Please load data first.")
                input("Press Enter to continue...")
                return
            
            self.ui.print("\n🔧 PREPROCESSING CONFIGURATION")
            self.ui.print("="*30)
            self.ui.print(f"Current validation size: {self.pipeline.test_size:.1%}")
            
            self.ui.print("\n1. Run preprocessing pipeline")
            self.ui.print("2. Change validation split size")
            self.ui.print("3. Manual preprocessing configuration")
            self.ui.print("0. Go Back")
            
            choice = self.ui.input("Enter choice (0-3)", default="0")
            
            if choice == "0":
                return
            elif choice == "1":
                success = self.pipeline.preprocessor.preprocess_data(self.pipeline)
                if success:
                    self.ui.print("✅ Preprocessing completed!")
                    self.ui.print(f"Training shape: {self.pipeline.X_train.shape}")
                    self.ui.print(f"Validation shape: {self.pipeline.X_test.shape}")
                input("Press Enter to continue...")
            elif choice == "2":
                self._change_validation_size()
            elif choice == "3":
                self._manual_preprocessing_menu()
            else:
                self.ui.print("❌ Invalid choice.")
                input("Press Enter to continue...")

    def handle_model_management_menu(self):
        """Handle model management menu."""
        while True:
            self.ui.print("\n🤖 MODEL MANAGEMENT")
            self.ui.print("="*30)
            
            enabled_count = sum(self.pipeline.available_models.values())
            self.ui.print(f"Currently enabled models: {enabled_count}")
            
            for model, enabled in self.pipeline.available_models.items():
                status = "✅" if enabled else "⏸️"
                self.ui.print(f"  {status} {model}")
            
            self.ui.print("\n1. Configure models")
            self.ui.print("2. View model documentation")
            self.ui.print("0. Go Back")
            
            choice = self.ui.input("Enter choice (0-2)", default="0")
            
            if choice == "0":
                return
            elif choice == "1":
                self._configure_models()
            elif choice == "2":
                self._show_model_documentation()
            else:
                self.ui.print("❌ Invalid choice.")
                input("Press Enter to continue...")

    def _show_data_exploration_menu(self):
        """Display data exploration menu options."""
        self.ui.print("\n🔍 ENHANCED DATA EXPLORATION & ANALYSIS")
        self.ui.print("="*50)
        
        if self.pipeline.dataset_analysis:
            quality_score = self.pipeline.dataset_analysis.get('data_quality', {}).get('quality_score', 0)
            self.ui.print(f"📊 Dataset Quality Score: {quality_score:.1f}/100")
        
        self.ui.print(f"📋 Dataset: {len(self.pipeline.train_data)} samples, {len(self.pipeline.train_data.columns)} columns")
        self.ui.print(f"🎯 Target: {self.pipeline.target_column} ({self.pipeline.problem_type})")
        
        self.ui.print(f"\n📈 ANALYSIS OPTIONS:")
        self.ui.print("1. 📊 Interactive Data Dashboard")
        self.ui.print("2. 📈 Statistical Analysis Suite")
        self.ui.print("3. 🎨 Visualization Gallery")
        self.ui.print("4. 🔍 Data Quality Assessment")
        self.ui.print("5. 🎯 Target Variable Deep Dive")
        self.ui.print("6. 🔗 Feature Relationships Analysis")
        self.ui.print("7. 📋 Comprehensive Data Report")
        self.ui.print("8. 💡 AI Insights & Recommendations")
        self.ui.print("9. 🔧 Feature Engineering Suggestions")
        self.ui.print("0. 🔙 Back to Main Menu")

    def _load_existing_dataset(self, dataset_info: Dict):
        """Load an existing dataset from database info."""
        self.ui.print(f"📂 Loading dataset: {dataset_info['name']}")
        
        if self.pipeline.db:
            try:
                experiments = self.pipeline.db.search_experiments(dataset_name=dataset_info['name'])
                if experiments:
                    latest_exp = experiments[0]
                    cursor = self.pipeline.db.connection.cursor()
                    cursor.execute(
                        'SELECT target_column, problem_type FROM experiments WHERE id = ?',
                        (latest_exp['id'],)
                    )
                    result = cursor.fetchone()
                    if result:
                        saved_target = result[0]
                        saved_problem_type = result[1]
                        
                        self.ui.print(f"📋 Found previous configuration:")
                        self.ui.print(f"   Target column: {saved_target}")
                        self.ui.print(f"   Problem type: {saved_problem_type}")
                        
                        use_saved = self.ui.confirm("Use saved configuration?", default=True)
                        if use_saved:
                            self.pipeline.target_column = saved_target
                            self.pipeline.problem_type = saved_problem_type
            except Exception as e:
                self.ui.print(f"⚠️ Could not load saved configuration: {e}")
        
        train_file = self.ui.input("📂 Enter training data file path", default=f"{dataset_info['name']}.csv")
        test_file = self.ui.input("📂 Enter test data file path (optional)", default="")
        
        if not self.pipeline.target_column:
            target_col = self.ui.input("🎯 Enter target column name", default="target")
            self.pipeline.target_column = target_col
        
        if test_file.strip() == "":
            test_file = None
        
        return self._perform_data_loading(train_file, test_file, self.pipeline.target_column, dataset_info['name'])

    def _load_new_dataset(self) -> bool:
        """Load a new dataset from file."""
        train_file = self.ui.input("📂 Enter training data file path", default="train.csv")
        test_file = self.ui.input("📂 Enter test data file path (optional)", default="")
        target_col = self.ui.input("🎯 Enter target column name", default="target")
        
        exp_name = self.ui.input("📝 Enter experiment name (optional)", default=self.pipeline.experiment_name)
        user_notes = self.ui.input("📋 Enter experiment notes (optional)", default="")
        
        if test_file.strip() == "":
            test_file = None
        
        return self._perform_data_loading(train_file, test_file, target_col, exp_name, user_notes)

    def _perform_data_loading(self, train_file: str, test_file: str = None, 
                             target_col: str = None, exp_name: str = None, 
                             user_notes: str = None) -> bool:
        """Perform the actual data loading process."""
        if exp_name:
            self.pipeline.experiment_name = exp_name
        if user_notes:
            self.pipeline.user_notes = user_notes
        
        import time
        self.pipeline.experiment_start_time = time.time()
        
        success = self.pipeline.data_ops.load_data(train_file, test_file, target_col, self.pipeline)
        
        if success:
            self.ui.print("✅ Data loaded successfully!")
            
            if self.pipeline.db:
                from pathlib import Path
                dataset_info = {
                    'experiment_name': self.pipeline.experiment_name,
                    'dataset_name': Path(train_file).stem,
                    'dataset_hash': self.pipeline.dataset_hash,
                    'problem_type': self.pipeline.problem_type,
                    'target_column': self.pipeline.target_column,
                    'n_samples': len(self.pipeline.train_data),
                    'n_features': len(self.pipeline.train_data.columns) - 1,
                    'train_size': 1 - self.pipeline.test_size,
                    'test_size': self.pipeline.test_size,
                    'cv_folds': self.pipeline.cv_folds,
                    'preprocessing_steps': [],
                    'user_notes': self.pipeline.user_notes
                }
                self.pipeline.experiment_id = self.pipeline.db.create_experiment(dataset_info)
            
            self.pipeline.data_ops.show_dataset_overview(
                self.pipeline.train_data, self.pipeline.target_column, self.pipeline.test_data)
            
            self._perform_initial_analysis()
            
            input("\nPress Enter to continue...")
            return True
        else:
            input("Press Enter to continue...")
            return False

    def _perform_initial_analysis(self):
        """Perform initial analysis with interactive follow-up options."""
        self.ui.print("\n🔍 Performing comprehensive dataset analysis...")
        
        self.pipeline.dataset_analysis = self.pipeline.analyzer.analyze_dataset_comprehensive(
            self.pipeline.train_data, self.pipeline.target_column
        )
        
        self._display_dataset_analysis()
        
        if self.pipeline.dataset_analysis and self.pipeline.dataset_analysis.get('recommendations'):
            self.ui.print("\n💡 Recommendations:")
            for i, rec in enumerate(self.pipeline.dataset_analysis['recommendations'], 1):
                self.ui.print(f"  {i}. {rec}")
            
            while True:
                self.ui.print("\nWhat would you like to do next?")
                self.ui.print("1. Follow recommendation: Feature scaling")
                self.ui.print("2. Follow recommendation: Handle missing values") 
                self.ui.print("3. Manual preprocessing configuration")
                self.ui.print("4. Proceed to data exploration")
                self.ui.print("5. Go to model configuration")
                self.ui.print("0. Return to main menu")
                
                choice = self.ui.input("Enter choice (0-5)", default="0")
                
                if choice == "0":
                    break
                elif choice == "1":
                    self._interactive_feature_scaling()
                    break
                elif choice == "2":
                    self._interactive_missing_values()
                    break
                elif choice == "3":
                    self._manual_preprocessing_menu()
                    break
                elif choice == "4":
                    self.handle_exploration_menu()
                    break
                elif choice == "5":
                    self.handle_model_management_menu()
                    break
                else:
                    self.ui.print("❌ Invalid choice.")

    def _display_dataset_analysis(self):
        """Display comprehensive dataset analysis results."""
        if not self.pipeline.dataset_analysis:
            return
        
        analysis = self.pipeline.dataset_analysis
        quality = analysis.get('data_quality', {})
        
        self.ui.print(f"\n🔍 DATA QUALITY ASSESSMENT")
        self.ui.print(f"Overall Quality Score: {quality.get('quality_score', 0):.1f}/100")
        
        if quality.get('issues'):
            self.ui.print("\n❌ Issues found:")
            for issue in quality['issues']:
                self.ui.print(f"  • {issue}")
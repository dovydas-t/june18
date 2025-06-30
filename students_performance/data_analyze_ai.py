# Load test data if provided
            if test_path:
                self.test_data = pd.read_csv(test_path)
                self.ui.print(f"[green]✅ Test data loaded: {self.test_data.shape}[/green]")
            
            # Set target column
            if target_column:
                self.target_column = target_column
            
            # Validate target column exists
            if self.target_column not in self.train_data.columns:
                self.ui.print(f"[red]❌ Target column '{self.target_column}' not found in training data![/red]")
                self.ui.print(f"Available columns: {list(self.train_data.columns)}")
                return False
            
            # Auto-detect problem type if not specified
            if self.problem_type == 'auto':
                self._detect_problem_type()
                self.ui.print(f"[blue]🔍 Auto-detected problem type: {self.problem_type}[/blue]")
            
            # Identify feature types
            self._identify_feature_types()
            
            return True
            
        except FileNotFoundError as e:
            self.ui.print(f"[red]❌ File not found: {e}[/red]")
            return False
        except pd.errors.EmptyDataError:
            self.ui.print(f"[red]❌ Empty CSV file provided[/red]")
            return False
        except Exception as e:
            self.ui.print(f"[red]❌ Error loading data: {str(e)}[/red]")
            return False
    
    def _detect_problem_type(self):
        """Enhanced problem type detection with detailed analysis."""
        if self.target_column not in self.train_data.columns:
            self.ui.print(f"[red]❌ Target column '{self.target_column}' not found[/red]")
            return
        
        target = self.train_data[self.target_column]
        
        # Remove missing values for analysis
        target_clean = target.dropna()
        
        if len(target_clean) == 0:
            self.ui.print("[red]❌ Target column contains only missing values[/red]")
            return
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(target_clean):
            unique_values = target_clean.nunique()
            total_values = len(target_clean)
            unique_ratio = unique_values / total_values
            
            # Decision logic for classification vs regression
            if unique_values <= 2:
                self.problem_type = 'classification'
                self.is_binary_classification = True
                self.ui.print(f"[blue]🎯 Binary classification detected (2 unique values)[/blue]")
            elif unique_values <= 20 and unique_ratio < 0.05:
                self.problem_type = 'classification'
                self.ui.print(f"[blue]🎯 Multi-class classification detected ({unique_values} classes)[/blue]")
            elif all(target_clean == target_clean.astype(int)):
                # Integer values - could be classification if reasonable number of classes
                if unique_values <= 50:
                    self.problem_type = 'classification'
                    self.ui.print(f"[blue]🎯 Classification detected ({unique_values} integer classes)[/blue]")
                else:
                    self.problem_type = 'regression'
                    self.ui.print(f"[blue]📈 Regression detected (many integer values)[/blue]")
            else:
                self.problem_type = 'regression'
                self.ui.print(f"[blue]📈 Regression detected (continuous values)[/blue]")
        else:
            # Non-numeric target - definitely classification
            self.problem_type = 'classification'
            unique_values = target_clean.nunique()
            if unique_values == 2:
                self.is_binary_classification = True
                self.ui.print(f"[blue]🎯 Binary classification detected (2 text classes)[/blue]")
            else:
                self.ui.print(f"[blue]🎯 Multi-class classification detected ({unique_values} text classes)[/blue]")
    
    def _identify_feature_types(self):
        """Identify and categorize feature types."""
        exclude_cols = ['Id', 'id', 'ID', self.target_column]
        all_columns = [col for col in self.train_data.columns if col not in exclude_cols]
        
        self.numerical_columns = []
        self.categorical_columns = []
        
        for col in all_columns:
            if pd.api.types.is_numeric_dtype(self.train_data[col]):
                self.numerical_columns.append(col)
            else:
                self.categorical_columns.append(col)
        
        self.feature_columns = all_columns
        
        self.ui.print(f"[blue]📊 Identified {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical features[/blue]")
    
    def run_pipeline_menu(self):
        """Enhanced pipeline execution with comprehensive tracking."""
        if self.train_data is None:
            self.ui.print("[red]❌ Please load data first![/red]")
            input("Press Enter to continue...")
            return
        
        self.ui.show_header("🚀 Complete ML Pipeline Execution", "Train multiple models and compare performance")
        
        # Show what will be executed
        enabled_models = [name for name, enabled in self.available_models.items() if enabled]
        
        if not enabled_models:
            self.ui.print("[red]❌ No models enabled! Please enable at least one model in Model Management.[/red]")
            input("Press Enter to continue...")
            return
        
        # Show execution plan
        if self.ui.console:
            plan_table = Table(title="🎯 Execution Plan", box=box.ROUNDED)
            plan_table.add_column("Step", style="cyan", width=20)
            plan_table.add_column("Description", style="white", width=50)
            plan_table.add_column("Status", style="green", width=15)
            
            steps = [
                ("Data Preprocessing", "Clean data, handle missing values, scale features", "Ready"),
                ("Feature Engineering", "Create interaction features, encode categories", "Ready"),
                ("Model Training", f"Train {len(enabled_models)} models with hyperparameter tuning", "Ready"),
                ("Model Evaluation", "Cross-validation and performance metrics", "Ready"),
                ("Results Analysis", "Compare models and generate insights", "Ready"),
                ("Database Logging", "Save experiment results to SQLite database", "Ready")
            ]
            
            for step, desc, status in steps:
                plan_table.add_row(step, desc, f"[green]✅ {status}[/green]")
            
            self.ui.console.print(plan_table)
        
        self.ui.print(f"\n[blue]📊 Will train {len(enabled_models)} models:[/blue]")
        for model in enabled_models:
            self.ui.print(f"  • {model}")
        
        if not self.ui.confirm("\n🚀 Start pipeline execution?"):
            return
        
        try:
            # Start comprehensive pipeline execution
            pipeline_start_time = time.time()
            
            # Step 1: Data Preprocessing
            self.ui.show_progress_with_steps([
                "🔧 Preprocessing data and handling missing values",
                "⚙️ Engineering features and encoding categories", 
                "📏 Scaling features and splitting data"
            ])
            
            self.preprocess_data()
            self.preprocessing_steps.append("Data cleaning and preprocessing completed")
            
            # Step 2: Model Training with detailed tracking
            self.ui.print("\n[bold blue]🤖 Starting model training with hyperparameter optimization...[/bold blue]")
            
            model_results = {}
            for i, model_name in enumerate(enabled_models, 1):
                self.ui.print(f"\n[cyan]Training {model_name} ({i}/{len(enabled_models)})...[/cyan]")
                
                model_start_time = time.time()
                
                try:
                    # Train model with error handling
                    if model_name == 'KNN':
                        model, params = self.train_knn_model()
                    elif model_name == 'SVM':
                        model, params = self.train_svm_model()
                    elif model_name == 'Decision Tree':
                        model, params = self.train_tree_model()
                    elif model_name == 'Linear Model':
                        model, params = self.train_linear_model()
                    elif model_name == 'Random Forest':
                        model, params = self.train_random_forest_model()
                    elif model_name == 'Extra Trees':
                        model, params = self.train_extra_trees_model()
                    elif model_name == 'Bagging':
                        model, params = self.train_bagging_model()
                    elif model_name == 'AdaBoost':
                        model, params = self.train_adaboost_model()
                    elif model_name == 'Hist Gradient Boosting':
                        model, params = self.train_hist_gradient_boosting_model()
                    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                        model, params = self.train_xgboost_model()
                    elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                        model, params = self.train_lightgbm_model()
                    else:
                        self.ui.print(f"[yellow]⚠️ Skipping {model_name} - not available[/yellow]")
                        continue
                    
                    model_training_time = time.time() - model_start_time
                    
                    # Store model and metadata
                    self.models[model_name] = model
                    self.model_metadata[model_name] = {
                        'training_time': model_training_time,
                        'hyperparameters': params,
                        'model_type': self._get_model_type(model_name)
                    }
                    
                    self.ui.print(f"[green]✅ {model_name} trained successfully in {model_training_time:.2f}s[/green]")
                    
                except Exception as e:
                    self.ui.print(f"[red]❌ Error training {model_name}: {str(e)}[/red]")
                    continue
            
            # Step 3: Model Evaluation
            self.ui.print("\n[bold blue]📊 Evaluating model performance...[/bold blue]")
            self.evaluate_models()
            
            # Step 4: Database Logging
            self.ui.print("\n[bold blue]🗄️ Saving results to database...[/bold blue]")
            self._log_experiment_results()
            
            # Step 5: Update experiment duration
            total_duration = time.time() - pipeline_start_time
            self.db.update_experiment_duration(self.experiment_id, total_duration)
            
            self.ui.print(f"\n[green]✅ Pipeline completed successfully in {total_duration:.2f} seconds![/green]")
            
            # Show quick results summary
            self._show_pipeline_summary()
            
        except Exception as e:
            self.ui.print(f"[red]❌ Pipeline failed: {str(e)}[/red]")
            import traceback
            self.ui.print(f"[dim]Debug info: {traceback.format_exc()}[/dim]")
        
        input("\nPress Enter to continue...")
    
    def _log_experiment_results(self):
        """Log all experiment results to the database."""
        if not self.experiment_id:
            return
        
        best_model_name = None
        best_score = -float('inf') if self.problem_type == 'classification' else float('inf')
        
        # Log each model's results
        for model_name, model in self.models.items():
            if model_name not in self.scores:
                continue
            
            # Determine if this is the best model
            if self.problem_type == 'classification':
                current_score = self.scores[model_name].get('Accuracy', 0)
                is_best = current_score > best_score
                if is_best:
                    best_score = current_score
                    best_model_name = model_name
            else:
                current_score = self.scores[model_name].get('RMSE', float('inf'))
                is_best = current_score < best_score
                if is_best:
                    best_score = current_score
                    best_model_name = model_name
            
            # Prepare model data
            model_data = {
                'model_name': model_name,
                'model_type': self.model_metadata[model_name]['model_type'],
                'hyperparameters': self.model_metadata[model_name]['hyperparameters'],
                'cv_score': self._get_cv_score(model_name),
                'training_time': self.model_metadata[model_name]['training_time'],
                'is_best_model': False  # Will be updated later
            }
            
            # Add performance scores
            if self.problem_type == 'classification':
                model_data['validation_score'] = self.scores[model_name].get('Accuracy', 0)
            else:
                model_data['validation_score'] = self.scores[model_name].get('RMSE', 0)
            
            # Log to database
            model_result_id = self.db.log_model_result(self.experiment_id, model_data)
            
            # Log metrics
            self.db.log_metrics(model_result_id, self.scores[model_name])
            
            # Log hyperparameters
            self.db.log_hyperparameters(
                model_result_id, 
                self.model_metadata[model_name]['hyperparameters']
            )
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.X_train.columns,
                    model.feature_importances_
                ))
                self.db.log_feature_importance(model_result_id, feature_importance)
        
        # Mark best model
        if best_model_name:
            best_model_result = self.db.conn.execute("""
                SELECT id FROM model_results 
                WHERE experiment_id = ? AND model_name = ?
            """, (self.experiment_id, best_model_name)).fetchone()
            
            if best_model_result:
                self.db.mark_best_model(self.experiment_id, best_model_result[0])
    
    def _get_cv_score(self, model_name: str) -> float:
        """Get cross-validation score for a model."""
        if model_name not in self.models:
            return 0.0
        
        try:
            model = self.models[model_name]
            scoring = self._get_scoring_metric()
            
            if self.problem_type == 'classification':
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=self.cv_folds, scoring=scoring)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                          cv=self.cv_folds, scoring=scoring)
            
            return cv_scores.mean()
        except:
            return 0.0
    
    def _get_model_type(self, model_name: str) -> str:
        """Get model type category."""
        type_mapping = {
            'KNN': 'instance_based',
            'SVM': 'kernel_based', 
            'Decision Tree': 'tree_based',
            'Linear Model': 'linear',
            'Random Forest': 'ensemble',
            'Extra Trees': 'ensemble',
            'Bagging': 'ensemble',
            'AdaBoost': 'boosting',
            'Hist Gradient Boosting': 'boosting',
            'XGBoost': 'boosting',
            'LightGBM': 'boosting'
        }
        return type_mapping.get(model_name, 'other')
    
    def _show_pipeline_summary(self):
        """Show comprehensive pipeline execution summary."""
        if not self.scores:
            return
        
        if self.ui.console:
            # Performance summary table
            summary_table = Table(title="🏆 Pipeline Execution Summary", box=box.ROUNDED)
            summary_table.add_column("Model", style="cyan", width=20)
            
            if self.problem_type == 'classification':
                summary_table.add_column("Accuracy", style="green", width=12)
                summary_table.add_column("F1-Score", style="yellow", width=12)
                summary_table.add_column("Precision", style="blue", width=12)
                summary_table.add_column("Recall", style="magenta", width=12)
                
                # Sort by accuracy
                sorted_models = sorted(self.scores.items(), key=lambda x: x[1].get('Accuracy', 0), reverse=True)
                
                for i, (model_name, scores) in enumerate(sorted_models):
                    rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                    summary_table.add_row(
                        f"{rank_emoji} {model_name}",
                        f"{scores.get('Accuracy', 0):.4f}",
                        f"{scores.get('F1', 0):.4f}",
                        f"{scores.get('Precision', 0):.4f}",
                        f"{scores.get('Recall', 0):.4f}"
                    )
            else:
                summary_table.add_column("RMSE", style="green", width=12)
                summary_table.add_column("R²", style="yellow", width=12)
                summary_table.add_column("MAE", style="blue", width=12)
                
                # Sort by RMSE (lower is better)
                sorted_models = sorted(self.scores.items(), key=lambda x: x[1].get('RMSE', float('inf')))
                
                for i, (model_name, scores) in enumerate(sorted_models):
                    rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                    summary_table.add_row(
                        f"{rank_emoji} {model_name}",
                        f"{scores.get('RMSE', 0):.4f}",
                        f"{scores.get('R2', 0):.4f}",
                        f"{scores.get('MAE', 0):.4f}"
                    )
            
            self.ui.console.print(summary_table)
            
            # Best model highlight
            if self.problem_type == 'classification':
                best_model = max(self.scores.keys(), key=lambda x: self.scores[x].get('Accuracy', 0))
                best_score = self.scores[best_model]['Accuracy']
                metric_name = "Accuracy"
            else:
                best_model = min(self.scores.keys(), key=lambda x: self.scores[x].get('RMSE', float('inf')))
                best_score = self.scores[best_model]['RMSE']
                metric_name = "RMSE"
            
            best_panel = Panel(
                f"[bold green]🏆 Best Model: {best_model}[/bold green]\n"
                f"[bold]{metric_name}: {best_score:.4f}[/bold]\n\n"
                f"Training Time: {self.model_metadata[best_model]['training_time']:.2f}s\n"
                f"Parameters: {len(self.model_metadata[best_model]['hyperparameters'])} tuned",
                title="🎯 Top Performer",
                border_style="green"
            )
            self.ui.console.print(best_panel)
        
        else:
            print("\nPipeline Summary:")
            for model_name, scores in self.scores.items():
                if self.problem_type == 'classification':
                    print(f"{model_name}: Accuracy = {scores.get('Accuracy', 0):.4f}")
                else:
                    print(f"{model_name}: RMSE = {scores.get('RMSE', 0):.4f}")
    
    def view_database_menu(self):
        """Enhanced database viewer with comprehensive experiment tracking."""
        while True:
            self.ui.show_header("🗄️ Experiment Database", "View and analyze your ML experiment history")
            
            options = [
                "📊 View All Experiments Summary",
                "🏆 View Best Models Across Experiments", 
                "📈 Compare Model Performance Trends",
                "🔍 Search Experiments by Criteria",
                "📋 View Detailed Experiment Report",
                "📉 Export Results to CSV",
                "🧹 Database Maintenance & Cleanup",
                "⬅️ Back to Main Menu"
            ]
            
            self.ui.show_menu("Database Options:", options)
            choice = self.ui.input("Enter your choice (1-8)", default="1")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._view_all_experiments()
                elif choice == 2:
                    self._view_best_models()
                elif choice == 3:
                    self._compare_performance_trends()
                elif choice == 4:
                    self._search_experiments()
                elif choice == 5:
                    self._view_detailed_experiment()
                elif choice == 6:
                    self._export_results()
                elif choice == 7:
                    self._database_maintenance()
                elif choice == 8:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            input("\nPress Enter to continue...")
    
    def _view_all_experiments(self):
        """View summary of all experiments."""
        try:
            cursor = self.db.conn.execute("""
                SELECT 
                    e.id,
                    e.experiment_name,
                    e.timestamp,
                    e.problem_type,
                    e.n_samples,
                    e.n_features,
                    COUNT(m.id) as model_count,
                    MAX(CASE WHEN m.is_best_model THEN 
                        CASE WHEN e.problem_type = 'classification' THEN m.validation_score
                             ELSE m.validation_score END
                    END) as best_score
                FROM experiments e
                LEFT JOIN model_results m ON e.id = m.experiment_id
                GROUP BY e.id
                ORDER BY e.timestamp DESC
                LIMIT 20
            """)
            
            experiments = cursor.fetchall()
            
            if not experiments:
                self.ui.print("[yellow]📝 No experiments found in database.[/yellow]")
                return
            
            if self.ui.console:
                table = Table(title="📊 Recent Experiments", box=box.ROUNDED)
                table.add_column("ID", style="cyan", width=5)
                table.add_column("Name", style="white", width=25)
                table.add_column("Date", style="yellow", width=12)
                table.add_column("Type", style="green", width=12)
                table.add_column("Samples", style="blue", width=8)
                table.add_column("Features", style="blue", width=8)
                table.add_column("Models", style="magenta", width=7)
                table.add_column("Best Score", style="red", width=10)
                
                for exp in experiments:
                    date_str = exp[2][:10] if exp[2] else "N/A"
                    best_score = f"{exp[7]:.4f}" if exp[7] else "N/A"
                    
                    table.add_row(
                        str(exp[0]),
                        exp[1][:22] + "..." if len(exp[1]) > 25 else exp[1],
                        date_str,
                        exp[3] or "N/A",
                        f"{exp[4]:,}" if exp[4] else "N/A",
                        str(exp[5]) if exp[5] else "N/A",
                        str(exp[6]),
                        best_score
                    )
                
                self.ui.console.print(table)
            else:
                print("\nRecent Experiments:")
                for exp in experiments:
                    print(f"ID {exp[0]}: {exp[1]} ({exp[2][:10]}) - {exp[6]} models")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error viewing experiments: {str(e)}[/red]")
    
    def _view_best_models(self):
        """View best models across all experiments."""
        try:
            best_models = self.db.get_best_models(limit=15)
            
            if not best_models:
                self.ui.print("[yellow]📝 No best models found in database.[/yellow]")
                return
            
            if self.ui.console:
                table = Table(title="🏆 Best Models Across All Experiments", box=box.ROUNDED)
                table.add_column("Experiment", style="cyan", width=20)
                table.add_column("Model", style="yellow", width=15)
                table.add_column("Type", style="green", width=12)
                table.add_column("CV Score", style="red", width=10)
                table.add_column("Val Score", style="blue", width=10)
                table.add_column("Date", style="white", width=12)
                
                for model in best_models:
                    date_str = model['timestamp'][:10] if model['timestamp'] else "N/A"
                    
                    table.add_row(
                        model['experiment_name'][:17] + "..." if len(model['experiment_name']) > 20 else model['experiment_name'],
                        model['model_name'],
                        model['problem_type'],
                        f"{model['cv_score']:.4f}" if model['cv_score'] else "N/A",
                        f"{model['validation_score']:.4f}" if model['validation_score'] else "N/A",
                        date_str
                    )
                
                self.ui.console.print(table)
            else:
                print("\nBest Models:")
                for model in best_models:
                    print(f"{model['model_name']} in {model['experiment_name']}: {model['validation_score']:.4f}")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error viewing best models: {str(e)}[/red]")


# Training methods for individual models
    def train_random_forest_model(self):
        """Train Random Forest with comprehensive parameter tuning."""
        print(f"🔄 Training Random Forest {'Classifier' if self.problem_type == 'classification' else 'Regressor'}...")
        
        if self.problem_type == 'classification':
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
        else:
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        grid_search = GridSearchCV(
            model, params,
            cv=min(3, self.cv_folds),
            scoring=self._get_scoring_metric(),
            n_jobs=1,  # Model already uses n_jobs
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best Random Forest params: {grid_search.best_params_}")
        print(f"   ✅ Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

    def train_extra_trees_model(self):
        """Train Extra Trees with comprehensive parameter tuning.""" 
        print(f"🔄 Training Extra Trees {'Classifier' if self.problem_type == 'classification' else 'Regressor'}...")
        
        if self.problem_type == 'classification':
            model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            model = ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1)
        
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            model, params,
            cv=3,
            scoring=self._get_scoring_metric(),
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best Extra Trees params: {grid_search.best_params_}")
        print(f"   ✅ Best Extra Trees CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

# Additional utility methods for the enhanced pipeline

def main():
    """Main execution function with enhanced error handling."""
    print("🚀 Starting Enhanced ML Pipeline v2.0 with Documentation & SQLite Tracking...")
    
    pipeline = EnhancedMLPipeline()
    
    try:
        pipeline.show_welcome()
        pipeline.main_menu()
    except KeyboardInterrupt:
        pipeline.ui.print("\n[yellow]👋 Pipeline interrupted by user.[/yellow]")
        pipeline.ui.print("[blue]💾 Your experiments have been saved to the database.[/blue]")
    except Exception as e:
        pipeline.ui.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")
        pipeline.ui.print("[yellow]Please report this issue if it persists.[/yellow]")
    finally:
        # Ensure database connection is closed
        if hasattr(pipeline, 'db'):
            pipeline.db.close()
            pipeline.ui.print("[dim]🗄️ Database connection closed.[/dim]")

if __name__ == "__main__":
    print("""
🎯 ENHANCED MACHINE LEARNING PIPELINE v2.0
==========================================

Choose your preferred mode:

1. 🖥️  Interactive Mode - Full terminal UI with guided workflow
2. 🚀 Quick Start Demo - Automated demo with sample data
3. 📚 Documentation & Help - Comprehensive guides and examples
4. 🗄️  Database Browser - View existing experiment results

""")
    
    try:
        mode = input("Enter mode (1-4): ").strip()
        
        if mode == "1":
            main()
        elif mode == "2":
            quick_start_demo()
        elif mode == "3":
            pipeline = EnhancedMLPipeline()
            pipeline.ui.show_interactive_help()
        elif mode == "4":
            pipeline = EnhancedMLPipeline()
            pipeline.view_database_menu()
        else:
            print("Starting interactive mode by default...")
            main()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Starting interactive mode...")
        main()

"""
🔧 INSTALLATION & SETUP GUIDE
=============================

REQUIRED DEPENDENCIES:
pip install pandas numpy matplotlib seaborn scikit-learn rich scipy

OPTIONAL DEPENDENCIES (for advanced models):
pip install xgboost lightgbm

DATABASE REQUIREMENTS:
SQLite (built into Python) - no additional installation needed

FEATURES SUMMARY:
================

🔍 COMPREHENSIVE DOCUMENTATION:
• Parameter explanations with mathematical foundations
• Model characteristics and use case recommendations  
• Preprocessing technique guidance with best practices
• Interactive help system with tooltips and legends

🗄️ SQLITE EXPERIMENT TRACKING:
• Automatic experiment logging with unique IDs
• Model performance metrics storage and comparison
• Hyperparameter search results tracking
• Feature importance analysis and storage
• Data preprocessing steps documentation
• Cross-experiment performance analysis

🤖 ENHANCED MODEL SUPPORT:
• 11 ML algorithms with detailed parameter tuning
• Automatic hyperparameter optimization with GridSearchCV
• Cross-validation with appropriate scoring metrics
• Model comparison and ranking systems
• Feature importance extraction and visualization

📊 ADVANCED VISUALIZATIONS:
• Model performance comparison charts
• Actual vs Predicted plots with confidence intervals
• Residual analysis for regression models
• Confusion matrices and ROC curves for classification
• Feature importance rankings and correlations
• Training time vs performance trade-off analysis

🔧 INTELLIGENT PREPROCESSING:
• Adaptive missing value imputation strategies
• Smart feature engineering with domain awareness
• Outlier detection and treatment recommendations
• Feature scaling optimization based on data distribution
• Categorical encoding with cardinality considerations

📈 COMPREHENSIVE ANALYSIS:
• Dataset quality assessment with scoring
• Column-by-column analysis with recommendations
• Statistical summaries with distribution analysis
• Correlation analysis with multicollinearity detection
• Data leakage prevention and validation

🎯 PRODUCTION READY:
• Automatic submission file generation
• Model serialization and deployment preparation
• Performance monitoring and comparison
• Experiment reproducibility with random state management
• Error handling and logging throughout pipeline

USAGE EXAMPLES:
==============

# Basic usage with interactive UI
python enhanced_ml_pipeline.py

# Programmatic usage
from enhanced_ml_pipeline import EnhancedMLPipeline

pipeline = EnhancedMLPipeline(
    problem_type='classification',
    target_column='target',
    experiment_name='My_Experiment'
)

# Load and process data
pipeline.load_data('train.csv', 'test.csv', 'target')
pipeline.preprocess_data()

# Train and evaluate models
pipeline.train_models()
pipeline.evaluate_models()

# Generate results
pipeline.create_visualizations()
submission = pipeline.generate_submission()

# View experiment database
results = pipeline.db.get_best_models()
summary = pipeline.db.get_experiment_summary(experiment_id)

BEST PRACTICES:
==============

1. 📊 Start with data exploration to understand your dataset
2. 🎯 Use appropriate evaluation metrics for your problem type  
3. 🔄 Always use cross-validation for reliable model selection
4. 📈 Monitor for overfitting by comparing train vs validation scores
5. 🔧 Apply consistent preprocessing across train/test splits
6. 💾 Keep detailed experiment logs for reproducibility
7. 🎛️ Start with default parameters, then optimize systematically
8. 📋 Document your findings and model insights
9. 🧪 Test different preprocessing strategies
10. 🏆 Focus on business metrics, not just statistical performance

TROUBLESHOOTING:
===============

• Database locked: Close other pipeline instances
• Memory errors: Reduce dataset size or use simpler models
• Import errors: Install missing dependencies with pip
• Slow performance: Disable complex models or reduce hyperparameter search space
• Visualization issues: Ensure matplotlib backend is properly configured

For detailed documentation and examples, run the interactive help system
or visit the comprehensive parameter documentation within the application.
"""
    # Main execution
    print("🚀 Starting Enhanced ML Pipeline v2.0...")
    
    pipeline = EnhancedMLPipeline()
    pipeline.show_welcome()
    
    try:
        pipeline.main_menu()
    except KeyboardInterrupt:
        pipeline.ui.print("\n[yellow]👋 Pipeline interrupted by user. Goodbye![/yellow]")
    except Exception as e:
        pipeline.ui.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")
    finally:
        # Close database connection
        if hasattr(pipeline, 'db'):
            pipeline.db.close()"""
🚀 Enhanced Machine Learning Pipeline Framework with Comprehensive Documentation & SQLite Tracking
======================================================================================================

Author: Advanced ML Pipeline Framework
Date: 2025
Version: 2.0

📋 COMPREHENSIVE FEATURE SET:
============================
✅ Interactive Terminal UI with Rich library and legends
✅ SQLite database for experiment tracking and performance logging
✅ Comprehensive parameter documentation with explanations
✅ 11 machine learning models with detailed hyperparameter tuning
✅ Smart data analysis and preprocessing recommendations
✅ Model performance comparison and visualization
✅ Automatic submission generation and model export
✅ In-depth tooltips and help system for all parameters
✅ Code clarity and user-friendly documentation

🤖 SUPPORTED MODELS WITH FULL DOCUMENTATION:
============================================
1. K-Nearest Neighbors (KNN) - Instance-based learning
2. Support Vector Machine (SVM/SVR) - Kernel-based classification/regression
3. Decision Tree - Tree-based interpretable model
4. Linear Model (Linear/Logistic Regression) - Linear baseline model
5. Random Forest - Ensemble of decision trees
6. Extra Trees - Extremely randomized trees
7. Bagging - Bootstrap aggregating ensemble
8. AdaBoost - Adaptive boosting sequential learning
9. Histogram Gradient Boosting - Memory-efficient gradient boosting
10. XGBoost - Extreme gradient boosting (optional)
11. LightGBM - Light gradient boosting machine (optional)

📊 DATABASE TRACKING FEATURES:
==============================
• Experiment metadata storage (timestamp, duration, dataset info)
• Parameter tracking for all models with explanations
• Performance metrics storage (accuracy, precision, recall, F1, RMSE, R², etc.)
• Hyperparameter optimization results logging
• Best model identification and comparison
• Data preprocessing steps tracking
• Feature engineering transformation logging
• Cross-validation results storage
• Model artifact paths and metadata
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                             ExtraTreesRegressor, ExtraTreesClassifier,
                             BaggingRegressor, BaggingClassifier,
                             AdaBoostRegressor, AdaBoostClassifier,
                             HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score)

# Terminal UI imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.tree import Tree
    from rich.markup import escape
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich library not available. Install with: pip install rich")

# Optional imports for advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Global configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATABASE_PATH = "ml_experiments.db"

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


class ExperimentDatabase:
    """
    SQLite database for tracking ML experiments, model performance, and metadata.
    Provides comprehensive storage and retrieval of experiment data.
    """
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Initialize database connection and create tables if they don't exist.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.create_tables()
    
    def create_tables(self):
        """Create database tables for experiment tracking."""
        
        # Main experiments table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dataset_name TEXT,
            dataset_hash TEXT,
            problem_type TEXT,
            target_column TEXT,
            n_samples INTEGER,
            n_features INTEGER,
            train_size REAL,
            test_size REAL,
            cv_folds INTEGER,
            preprocessing_steps TEXT,
            duration_seconds REAL,
            user_notes TEXT
        )
        """)
        
        # Model results table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_name TEXT NOT NULL,
            model_type TEXT,
            hyperparameters TEXT,
            cv_score REAL,
            cv_std REAL,
            train_score REAL,
            validation_score REAL,
            training_time REAL,
            prediction_time REAL,
            model_size_mb REAL,
            feature_importance TEXT,
            is_best_model BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        """)
        
        # Performance metrics table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_result_id INTEGER,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            metric_type TEXT,
            dataset_split TEXT,
            FOREIGN KEY (model_result_id) REFERENCES model_results (id)
        )
        """)
        
        # Hyperparameter search table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperparameter_search (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_result_id INTEGER,
            parameter_name TEXT NOT NULL,
            parameter_value TEXT,
            search_method TEXT,
            search_space TEXT,
            is_best BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (model_result_id) REFERENCES model_results (id)
        )
        """)
        
        # Feature importance table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_importance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_result_id INTEGER,
            feature_name TEXT NOT NULL,
            importance_value REAL,
            importance_rank INTEGER,
            FOREIGN KEY (model_result_id) REFERENCES model_results (id)
        )
        """)
        
        self.conn.commit()
    
    def create_experiment(self, experiment_data: Dict[str, Any]) -> int:
        """
        Create new experiment entry and return experiment ID.
        
        Args:
            experiment_data: Dictionary containing experiment metadata
            
        Returns:
            experiment_id: ID of created experiment
        """
        cursor = self.conn.execute("""
        INSERT INTO experiments (
            experiment_name, dataset_name, dataset_hash, problem_type, target_column,
            n_samples, n_features, train_size, test_size, cv_folds, 
            preprocessing_steps, user_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_data.get('experiment_name'),
            experiment_data.get('dataset_name'),
            experiment_data.get('dataset_hash'),
            experiment_data.get('problem_type'),
            experiment_data.get('target_column'),
            experiment_data.get('n_samples'),
            experiment_data.get('n_features'),
            experiment_data.get('train_size'),
            experiment_data.get('test_size'),
            experiment_data.get('cv_folds'),
            json.dumps(experiment_data.get('preprocessing_steps', [])),
            experiment_data.get('user_notes')
        ))
        
        experiment_id = cursor.lastrowid
        self.conn.commit()
        return experiment_id
    
    def log_model_result(self, experiment_id: int, model_data: Dict[str, Any]) -> int:
        """
        Log model training results.
        
        Args:
            experiment_id: ID of associated experiment
            model_data: Dictionary containing model results and metadata
            
        Returns:
            model_result_id: ID of created model result entry
        """
        cursor = self.conn.execute("""
        INSERT INTO model_results (
            experiment_id, model_name, model_type, hyperparameters, cv_score, cv_std,
            train_score, validation_score, training_time, prediction_time,
            model_size_mb, feature_importance, is_best_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            model_data.get('model_name'),
            model_data.get('model_type'),
            json.dumps(model_data.get('hyperparameters', {})),
            model_data.get('cv_score'),
            model_data.get('cv_std'),
            model_data.get('train_score'),
            model_data.get('validation_score'),
            model_data.get('training_time'),
            model_data.get('prediction_time'),
            model_data.get('model_size_mb'),
            json.dumps(model_data.get('feature_importance', {})),
            model_data.get('is_best_model', False)
        ))
        
        model_result_id = cursor.lastrowid
        self.conn.commit()
        return model_result_id
    
    def log_metrics(self, model_result_id: int, metrics: Dict[str, float], 
                   dataset_split: str = 'validation'):
        """
        Log performance metrics for a model.
        
        Args:
            model_result_id: ID of model result entry
            metrics: Dictionary of metric name -> value
            dataset_split: Which dataset split these metrics are from
        """
        for metric_name, metric_value in metrics.items():
            self.conn.execute("""
            INSERT INTO performance_metrics (
                model_result_id, metric_name, metric_value, metric_type, dataset_split
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                model_result_id,
                metric_name,
                float(metric_value),
                self._get_metric_type(metric_name),
                dataset_split
            ))
        
        self.conn.commit()
    
    def log_hyperparameters(self, model_result_id: int, hyperparameters: Dict[str, Any], 
                           search_method: str = 'grid_search', is_best: bool = True):
        """
        Log hyperparameters used for a model.
        
        Args:
            model_result_id: ID of model result entry
            hyperparameters: Dictionary of parameter name -> value
            search_method: Method used for hyperparameter search
            is_best: Whether these are the best parameters found
        """
        for param_name, param_value in hyperparameters.items():
            self.conn.execute("""
            INSERT INTO hyperparameter_search (
                model_result_id, parameter_name, parameter_value, search_method, is_best
            ) VALUES (?, ?, ?, ?, ?)
            """, (
                model_result_id,
                param_name,
                str(param_value),
                search_method,
                is_best
            ))
        
        self.conn.commit()
    
    def log_feature_importance(self, model_result_id: int, feature_importance: Dict[str, float]):
        """
        Log feature importance scores.
        
        Args:
            model_result_id: ID of model result entry
            feature_importance: Dictionary of feature name -> importance score
        """
        # Sort by importance and add ranks
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (feature_name, importance_value) in enumerate(sorted_features, 1):
            self.conn.execute("""
            INSERT INTO feature_importance (
                model_result_id, feature_name, importance_value, importance_rank
            ) VALUES (?, ?, ?, ?)
            """, (
                model_result_id,
                feature_name,
                float(importance_value),
                rank
            ))
        
        self.conn.commit()
    
    def update_experiment_duration(self, experiment_id: int, duration_seconds: float):
        """Update experiment duration."""
        self.conn.execute("""
        UPDATE experiments SET duration_seconds = ? WHERE id = ?
        """, (duration_seconds, experiment_id))
        self.conn.commit()
    
    def mark_best_model(self, experiment_id: int, model_result_id: int):
        """Mark a model as the best for an experiment."""
        # First, unmark all models for this experiment
        self.conn.execute("""
        UPDATE model_results SET is_best_model = FALSE WHERE experiment_id = ?
        """, (experiment_id,))
        
        # Then mark the specified model as best
        self.conn.execute("""
        UPDATE model_results SET is_best_model = TRUE WHERE id = ?
        """, (model_result_id,))
        
        self.conn.commit()
    
    def get_best_models(self, limit: int = 10) -> List[Dict]:
        """Get best models across all experiments."""
        cursor = self.conn.execute("""
        SELECT 
            e.experiment_name,
            e.timestamp,
            e.problem_type,
            m.model_name,
            m.cv_score,
            m.validation_score,
            m.hyperparameters,
            m.training_time
        FROM model_results m
        JOIN experiments e ON m.experiment_id = e.id
        WHERE m.is_best_model = TRUE
        ORDER BY m.cv_score DESC
        LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_experiment_summary(self, experiment_id: int) -> Dict:
        """Get comprehensive summary of an experiment."""
        # Get experiment details
        exp_cursor = self.conn.execute("""
        SELECT * FROM experiments WHERE id = ?
        """, (experiment_id,))
        experiment = dict(exp_cursor.fetchone())
        
        # Get model results
        models_cursor = self.conn.execute("""
        SELECT 
            m.*,
            GROUP_CONCAT(pm.metric_name || ':' || pm.metric_value) as metrics
        FROM model_results m
        LEFT JOIN performance_metrics pm ON m.id = pm.model_result_id
        WHERE m.experiment_id = ?
        GROUP BY m.id
        ORDER BY m.cv_score DESC
        """, (experiment_id,))
        
        models = [dict(row) for row in models_cursor.fetchall()]
        
        return {
            'experiment': experiment,
            'models': models,
            'total_models': len(models),
            'best_score': max([m['cv_score'] for m in models]) if models else None
        }
    
    def get_model_comparison(self, experiment_ids: List[int] = None) -> pd.DataFrame:
        """Get model comparison data as DataFrame."""
        where_clause = ""
        params = []
        
        if experiment_ids:
            placeholders = ",".join(["?" for _ in experiment_ids])
            where_clause = f"WHERE e.id IN ({placeholders})"
            params = experiment_ids
        
        query = f"""
        SELECT 
            e.experiment_name,
            e.problem_type,
            m.model_name,
            m.cv_score,
            m.cv_std,
            m.validation_score,
            m.training_time,
            m.is_best_model,
            e.timestamp
        FROM model_results m
        JOIN experiments e ON m.experiment_id = e.id
        {where_clause}
        ORDER BY e.timestamp DESC, m.cv_score DESC
        """
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def _get_metric_type(self, metric_name: str) -> str:
        """Determine metric type based on metric name."""
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roc_auc']
        regression_metrics = ['rmse', 'mae', 'r2', 'mse', 'mape']
        
        metric_lower = metric_name.lower()
        
        if any(cm in metric_lower for cm in classification_metrics):
            return 'classification'
        elif any(rm in metric_lower for rm in regression_metrics):
            return 'regression'
        else:
            return 'other'

    def _compare_performance_trends(self):
        """Compare model performance trends across experiments."""
        try:
            # Get comparison data
            comparison_df = self.db.get_model_comparison()
            
            if comparison_df.empty:
                self.ui.print("[yellow]📝 No model results found for comparison.[/yellow]")
                return
            
            if self.ui.console:
                # Performance trends table
                table = Table(title="📈 Model Performance Trends", box=box.ROUNDED)
                table.add_column("Model", style="cyan", width=15)
                table.add_column("Experiments", style="yellow", width=12)
                table.add_column("Avg CV Score", style="green", width=12)
                table.add_column("Best Score", style="red", width=12)
                table.add_column("Avg Training Time", style="blue", width=15)
                
                # Group by model name and calculate statistics
                model_stats = comparison_df.groupby('model_name').agg({
                    'cv_score': ['count', 'mean', 'max'],
                    'training_time': 'mean'
                }).round(4)
                
                for model_name in model_stats.index:
                    exp_count = int(model_stats.loc[model_name, ('cv_score', 'count')])
                    avg_score = model_stats.loc[model_name, ('cv_score', 'mean')]
                    best_score = model_stats.loc[model_name, ('cv_score', 'max')]
                    avg_time = model_stats.loc[model_name, ('training_time', 'mean')]
                    
                    table.add_row(
                        model_name,
                        str(exp_count),
                        f"{avg_score:.4f}" if not pd.isna(avg_score) else "N/A",
                        f"{best_score:.4f}" if not pd.isna(best_score) else "N/A",
                        f"{avg_time:.2f}s" if not pd.isna(avg_time) else "N/A"
                    )
                
                self.ui.console.print(table)
                
                # Show trend insights
                insights = []
                if len(comparison_df) > 0:
                    best_model = comparison_df.loc[comparison_df['cv_score'].idxmax(), 'model_name']
                    insights.append(f"🏆 Best performing model overall: {best_model}")
                    
                    fastest_model = comparison_df.loc[comparison_df['training_time'].idxmin(), 'model_name']
                    insights.append(f"⚡ Fastest training model: {fastest_model}")
                    
                    most_used = comparison_df['model_name'].value_counts().index[0]
                    insights.append(f"📊 Most frequently used model: {most_used}")
                
                if insights:
                    insights_panel = Panel(
                        "\n".join([f"• {insight}" for insight in insights]),
                        title="💡 Performance Insights",
                        border_style="blue"
                    )
                    self.ui.console.print(insights_panel)
            else:
                print("\nModel Performance Trends:")
                model_stats = comparison_df.groupby('model_name')['cv_score'].agg(['count', 'mean', 'max'])
                for model_name, stats in model_stats.iterrows():
                    print(f"{model_name}: {stats['count']} experiments, avg: {stats['mean']:.4f}, best: {stats['max']:.4f}")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error comparing performance trends: {str(e)}[/red]")

    def _search_experiments(self):
        """Search experiments by various criteria."""
        try:
            self.ui.print("\n[bold]🔍 Search Experiments[/bold]")
            
            search_options = [
                "📅 Search by date range",
                "🤖 Search by model name",
                "📊 Search by problem type",
                "🎯 Search by performance threshold",
                "📝 Search by experiment name"
            ]
            
            self.ui.show_menu("Search Options:", search_options)
            choice = self.ui.input("Enter search type (1-5)", default="1")
            
            where_conditions = []
            params = []
            
            try:
                choice = int(choice)
                if choice == 1:  # Date range
                    start_date = self.ui.input("Enter start date (YYYY-MM-DD)", default="2024-01-01")
                    end_date = self.ui.input("Enter end date (YYYY-MM-DD)", default="2025-12-31")
                    where_conditions.append("e.timestamp BETWEEN ? AND ?")
                    params.extend([start_date, end_date + " 23:59:59"])
                    
                elif choice == 2:  # Model name
                    model_name = self.ui.input("Enter model name")
                    where_conditions.append("m.model_name LIKE ?")
                    params.append(f"%{model_name}%")
                    
                elif choice == 3:  # Problem type
                    problem_type = self.ui.input("Enter problem type (classification/regression)")
                    where_conditions.append("e.problem_type = ?")
                    params.append(problem_type)
                    
                elif choice == 4:  # Performance threshold
                    threshold = float(self.ui.input("Enter minimum CV score", default="0.8"))
                    where_conditions.append("m.cv_score >= ?")
                    params.append(threshold)
                    
                elif choice == 5:  # Experiment name
                    exp_name = self.ui.input("Enter experiment name (partial match)")
                    where_conditions.append("e.experiment_name LIKE ?")
                    params.append(f"%{exp_name}%")
                
                # Build and execute query
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = f"""
                SELECT 
                    e.id,
                    e.experiment_name,
                    e.timestamp,
                    e.problem_type,
                    m.model_name,
                    m.cv_score,
                    m.validation_score,
                    m.is_best_model
                FROM experiments e
                LEFT JOIN model_results m ON e.id = m.experiment_id
                WHERE {where_clause}
                ORDER BY e.timestamp DESC, m.cv_score DESC
                LIMIT 50
                """
                
                cursor = self.db.conn.execute(query, params)
                results = cursor.fetchall()
                
                if not results:
                    self.ui.print("[yellow]📝 No experiments found matching your criteria.[/yellow]")
                    return
                
                # Display results
                if self.ui.console:
                    table = Table(title=f"🔍 Search Results ({len(results)} found)", box=box.ROUNDED)
                    table.add_column("ID", style="cyan", width=5)
                    table.add_column("Experiment", style="white", width=20)
                    table.add_column("Date", style="yellow", width=12)
                    table.add_column("Type", style="green", width=12)
                    table.add_column("Model", style="blue", width=15)
                    table.add_column("CV Score", style="red", width=10)
                    table.add_column("Best", style="magenta", width=8)
                    
                    for result in results:
                        date_str = result[2][:10] if result[2] else "N/A"
                        cv_score = f"{result[5]:.4f}" if result[5] else "N/A"
                        is_best = "🏆" if result[7] else ""
                        
                        table.add_row(
                            str(result[0]),
                            result[1][:17] + "..." if result[1] and len(result[1]) > 20 else result[1] or "N/A",
                            date_str,
                            result[3] or "N/A",
                            result[4] or "N/A",
                            cv_score,
                            is_best
                        )
                    
                    self.ui.console.print(table)
                else:
                    print(f"\nSearch Results ({len(results)} found):")
                    for result in results:
                        print(f"ID {result[0]}: {result[1]} - {result[4]} ({result[5]:.4f})")
            
            except ValueError:
                self.ui.print("[red]❌ Invalid input format.[/red]")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error searching experiments: {str(e)}[/red]")

    def _view_detailed_experiment(self):
        """View detailed information about a specific experiment."""
        try:
            exp_id = int(self.ui.input("Enter experiment ID to view"))
            
            summary = self.db.get_experiment_summary(exp_id)
            
            if not summary['experiment']:
                self.ui.print(f"[red]❌ Experiment {exp_id} not found.[/red]")
                return
            
            exp = summary['experiment']
            models = summary['models']
            
            if self.ui.console:
                # Experiment details
                exp_panel = Panel(
                    f"[bold]Name:[/bold] {exp['experiment_name']}\n"
                    f"[bold]Date:[/bold] {exp['timestamp']}\n"
                    f"[bold]Problem Type:[/bold] {exp['problem_type']}\n"
                    f"[bold]Dataset:[/bold] {exp['dataset_name']} ({exp['n_samples']} samples, {exp['n_features']} features)\n"
                    f"[bold]Duration:[/bold] {exp['duration_seconds']:.2f}s" if exp['duration_seconds'] else "Duration: N/A" + "\n"
                    f"[bold]Notes:[/bold] {exp['user_notes'] or 'None'}",
                    title=f"📋 Experiment {exp_id} Details",
                    border_style="blue"
                )
                self.ui.console.print(exp_panel)
                
                # Models table
                if models:
                    model_table = Table(title="🤖 Model Results", box=box.ROUNDED)
                    model_table.add_column("Model", style="cyan")
                    model_table.add_column("CV Score", style="green")
                    model_table.add_column("Val Score", style="yellow")
                    model_table.add_column("Training Time", style="blue")
                    model_table.add_column("Best", style="red")
                    
                    for model in models:
                        model_table.add_row(
                            model['model_name'],
                            f"{model['cv_score']:.4f}" if model['cv_score'] else "N/A",
                            f"{model['validation_score']:.4f}" if model['validation_score'] else "N/A",
                            f"{model['training_time']:.2f}s" if model['training_time'] else "N/A",
                            "🏆" if model['is_best_model'] else ""
                        )
                    
                    self.ui.console.print(model_table)
            else:
                print(f"\nExperiment {exp_id} Details:")
                print(f"Name: {exp['experiment_name']}")
                print(f"Date: {exp['timestamp']}")
                print(f"Problem Type: {exp['problem_type']}")
                print(f"Models trained: {len(models)}")
        
        except ValueError:
            self.ui.print("[red]❌ Please enter a valid experiment ID.[/red]")
        except Exception as e:
            self.ui.print(f"[red]❌ Error viewing experiment: {str(e)}[/red]")

    def _export_results(self):
        """Export experiment results to CSV."""
        try:
            # Get all model results
            query = """
            SELECT 
                e.experiment_name,
                e.timestamp,
                e.problem_type,
                e.dataset_name,
                m.model_name,
                m.model_type,
                m.cv_score,
                m.validation_score,
                m.training_time,
                m.is_best_model
            FROM model_results m
            JOIN experiments e ON m.experiment_id = e.id
            ORDER BY e.timestamp DESC
            """
            
            df = pd.read_sql_query(query, self.db.conn)
            
            if df.empty:
                self.ui.print("[yellow]📝 No results to export.[/yellow]")
                return
            
            filename = self.ui.input("Enter filename for export", default="ml_experiment_results.csv")
            
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            df.to_csv(filename, index=False)
            
            self.ui.print(f"[green]✅ Results exported to {filename}[/green]")
            self.ui.print(f"[blue]📊 Exported {len(df)} model results from {df['experiment_name'].nunique()} experiments[/blue]")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error exporting results: {str(e)}[/red]")

    def _database_maintenance(self):
        """Database maintenance and cleanup operations."""
        try:
            self.ui.print("\n[bold]🧹 Database Maintenance[/bold]")
            
            maintenance_options = [
                "📊 View database statistics",
                "🗑️ Delete old experiments (>30 days)",
                "🧹 Clean up incomplete experiments",
                "📦 Vacuum database (optimize storage)",
                "🔄 Rebuild database indexes",
                "📋 Export database backup"
            ]
            
            self.ui.show_menu("Maintenance Options:", maintenance_options)
            choice = self.ui.input("Enter choice (1-6)", default="1")
            
            try:
                choice = int(choice)
                
                if choice == 1:  # Database statistics
                    # Get database statistics
                    cursor = self.db.conn.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM experiments) as total_experiments,
                        (SELECT COUNT(*) FROM model_results) as total_models,
                        (SELECT COUNT(*) FROM performance_metrics) as total_metrics,
                        (SELECT MIN(timestamp) FROM experiments) as oldest_experiment,
                        (SELECT MAX(timestamp) FROM experiments) as newest_experiment
                    """)
                    
                    stats = cursor.fetchone()
                    
                    if self.ui.console:
                        stats_panel = Panel(
                            f"📊 Total Experiments: {stats[0]}\n"
                            f"🤖 Total Model Results: {stats[1]}\n"
                            f"📈 Total Metrics Recorded: {stats[2]}\n"
                            f"📅 Date Range: {stats[3][:10] if stats[3] else 'N/A'} to {stats[4][:10] if stats[4] else 'N/A'}",
                            title="🗄️ Database Statistics",
                            border_style="green"
                        )
                        self.ui.console.print(stats_panel)
                    else:
                        print(f"Total Experiments: {stats[0]}")
                        print(f"Total Model Results: {stats[1]}")
                        print(f"Total Metrics: {stats[2]}")
                
                elif choice == 2:  # Delete old experiments
                    if self.ui.confirm("⚠️ Delete experiments older than 30 days?"):
                        cursor = self.db.conn.execute("""
                        DELETE FROM experiments 
                        WHERE timestamp < datetime('now', '-30 days')
                        """)
                        deleted = cursor.rowcount
                        self.db.conn.commit()
                        self.ui.print(f"[green]✅ Deleted {deleted} old experiments[/green]")
                
                elif choice == 3:  # Clean incomplete experiments
                    cursor = self.db.conn.execute("""
                    DELETE FROM experiments 
                    WHERE id NOT IN (SELECT DISTINCT experiment_id FROM model_results)
                    """)
                    deleted = cursor.rowcount
                    self.db.conn.commit()
                    self.ui.print(f"[green]✅ Cleaned {deleted} incomplete experiments[/green]")
                
                elif choice == 4:  # Vacuum database
                    self.db.conn.execute("VACUUM")
                    self.ui.print("[green]✅ Database vacuumed and optimized[/green]")
                
                elif choice == 5:  # Rebuild indexes
                    # Drop and recreate indexes
                    self.db.conn.execute("DROP INDEX IF EXISTS idx_experiments_timestamp")
                    self.db.conn.execute("DROP INDEX IF EXISTS idx_model_results_experiment")
                    self.db.conn.execute("CREATE INDEX idx_experiments_timestamp ON experiments(timestamp)")
                    self.db.conn.execute("CREATE INDEX idx_model_results_experiment ON model_results(experiment_id)")
                    self.db.conn.commit()
                    self.ui.print("[green]✅ Database indexes rebuilt[/green]")
                
                elif choice == 6:  # Export backup
                    backup_name = f"ml_experiments_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                    
                    # Simple file copy for SQLite backup
                    import shutil
                    shutil.copy2(self.db.db_path, backup_name)
                    self.ui.print(f"[green]✅ Database backup created: {backup_name}[/green]")
            
            except ValueError:
                self.ui.print("[red]❌ Invalid choice.[/red]")
        
        except Exception as e:
            self.ui.print(f"[red]❌ Error in database maintenance: {str(e)}[/red]")


    def close(self):
            """Close database connection."""
            self.conn.close()


class EnhancedTerminalUI:
    """
    Enhanced Terminal UI with comprehensive legends, tooltips, and help system.
    Provides user-friendly interface with detailed explanations for all functionality.
    """
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        # Initialize parameter documentation
        self.param_docs = ParameterDocumentation()
        self.model_docs = self.param_docs.get_model_docs()
        self.preprocessing_docs = self.param_docs.get_preprocessing_docs()
        self.metrics_docs = self.param_docs.get_evaluation_metrics_docs()
    
    def print(self, *args, **kwargs):
        """Enhanced print with Rich formatting."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)
    
    def input(self, prompt, **kwargs):
        """Enhanced input with Rich formatting."""
        if self.console:
            return Prompt.ask(prompt, **kwargs)
        else:
            return input(f"{prompt}: ")
    
    def confirm(self, prompt, default=True):
        """Enhanced confirmation dialog."""
        if self.console:
            return Confirm.ask(prompt, default=default)
        else:
            response = input(f"{prompt} (y/n): ").lower()
            return response in ['y', 'yes'] if response else default
    
    def show_header(self, title, subtitle=None):
        """Show enhanced header with optional subtitle."""
        if self.console:
            header_text = title
            if subtitle:
                header_text += f"\n[dim]{subtitle}[/dim]"
            
            self.console.print(Panel(
                Align.center(header_text),
                style="bold blue",
                box=box.DOUBLE,
                padding=(1, 2)
            ))
        else:
            print(f"\n{'='*60}\n{title}\n{'='*60}")
            if subtitle:
                print(f"{subtitle}\n")
    
    def show_legend(self, legend_type: str):
        """Show comprehensive legends for different UI elements."""
        legends = {
            "main_menu": {
                "title": "🏠 Main Menu Legend",
                "items": [
                    ("📁 Load Data", "Import training and test datasets, configure target column"),
                    ("🔍 Explore Data", "Analyze data quality, visualize distributions, get preprocessing recommendations"),
                    ("🔧 Configure Preprocessing", "Set up data cleaning, scaling, and feature engineering"),
                    ("🤖 Model Management", "Select models, view parameter explanations, configure hyperparameters"),
                    ("🚀 Run Pipeline", "Execute complete ML workflow with selected models"),
                    ("📊 View Results", "Compare model performance, analyze metrics, view visualizations"),
                    ("💾 Generate Submission", "Create predictions with best model for competition submission"),
                    ("📚 Help & Documentation", "Access comprehensive guides and parameter explanations")
                ]
            },
            
            "model_parameters": {
                "title": "🤖 Model Parameters Legend",
                "items": [
                    ("Regularization", "Controls overfitting by penalizing model complexity"),
                    ("Learning Rate", "Step size for gradient descent optimization"),
                    ("Tree Depth", "Maximum levels in decision tree (deeper = more complex)"),
                    ("Cross-Validation", "Technique to estimate model performance on unseen data"),
                    ("Grid Search", "Systematic search through hyperparameter combinations"),
                    ("Feature Importance", "Measure of how much each feature contributes to predictions")
                ]
            },
            
            "metrics": {
                "title": "📊 Evaluation Metrics Legend",
                "items": [
                    ("Accuracy", "Proportion of correct predictions (good for balanced data)"),
                    ("Precision", "Of positive predictions, how many were correct?"),
                    ("Recall", "Of actual positives, how many were found?"),
                    ("F1-Score", "Harmonic mean of precision and recall"),
                    ("RMSE", "Root Mean Square Error (lower is better for regression)"),
                    ("R²", "Coefficient of determination (higher is better, max = 1.0)")
                ]
            },
            
            "preprocessing": {
                "title": "🔧 Preprocessing Legend",
                "items": [
                    ("Missing Values", "Handle gaps in data (imputation strategies)"),
                    ("Feature Scaling", "Normalize feature ranges (StandardScaler, MinMaxScaler)"),
                    ("Outlier Detection", "Identify and handle extreme values"),
                    ("Feature Engineering", "Create new features from existing ones"),
                    ("Encoding", "Convert categorical variables to numerical format"),
                    ("Data Splitting", "Divide data into training and validation sets")
                ]
            }
        }
        
        if legend_type in legends:
            legend = legends[legend_type]
            
            if self.console:
                table = Table(title=legend["title"], box=box.ROUNDED, show_header=False)
                table.add_column("Term", style="cyan", width=20)
                table.add_column("Explanation", style="white")
                
                for term, explanation in legend["items"]:
                    table.add_row(term, explanation)
                
                self.console.print(table)
            else:
                print(f"\n{legend['title']}")
                print("-" * 50)
                for term, explanation in legend["items"]:
                    print(f"{term}: {explanation}")
    
    def show_model_documentation(self, model_name: str):
        """Show comprehensive documentation for a specific model."""
        if model_name not in self.model_docs:
            self.print(f"[red]Documentation not available for {model_name}[/red]")
            return
        
        doc = self.model_docs[model_name]
        
        if self.console:
            # Main information panel
            self.console.print(Panel(
                f"[bold]{doc['name']}[/bold]\n\n"
                f"{doc['description']}\n\n"
                f"[bold cyan]Mathematical Foundation:[/bold cyan]\n{doc['mathematical_foundation']}\n\n"
                f"[bold green]Pros:[/bold green] {', '.join(doc['pros'])}\n"
                f"[bold red]Cons:[/bold red] {', '.join(doc['cons'])}",
                title=f"📖 {model_name} Documentation",
                border_style="blue"
            ))
            
            # Parameters table
            param_table = Table(title="🔧 Parameters", box=box.ROUNDED)
            param_table.add_column("Parameter", style="cyan", width=15)
            param_table.add_column("Description", style="white", width=30)
            param_table.add_column("Range/Options", style="yellow", width=20)
            param_table.add_column("Effect", style="green", width=25)
            
            for param_name, param_info in doc['parameters'].items():
                param_table.add_row(
                    param_name,
                    param_info['description'],
                    param_info.get('range', str(param_info.get('options', 'N/A'))),
                    param_info['effect']
                )
            
            self.console.print(param_table)
            
            # Additional info
            self.console.print(Panel(
                f"[bold cyan]Use Cases:[/bold cyan] {', '.join(doc['use_cases'])}\n\n"
                f"[bold yellow]Preprocessing Requirements:[/bold yellow]\n" +
                '\n'.join([f"• {req}" for req in doc['preprocessing_requirements']]) + "\n\n"
                f"[bold magenta]Computational Complexity:[/bold magenta] {doc['complexity']}",
                title="📋 Additional Information",
                border_style="green"
            ))
        else:
            print(f"\n{doc['name']} Documentation")
            print("=" * 50)
            print(f"Description: {doc['description']}")
            print(f"Mathematical Foundation: {doc['mathematical_foundation']}")
            print(f"Use Cases: {', '.join(doc['use_cases'])}")
            print(f"Pros: {', '.join(doc['pros'])}")
            print(f"Cons: {', '.join(doc['cons'])}")
            print(f"Complexity: {doc['complexity']}")
    
    def show_preprocessing_help(self, topic: str):
        """Show help for preprocessing topics."""
        if topic not in self.preprocessing_docs:
            self.print(f"[red]Documentation not available for {topic}[/red]")
            return
        
        doc = self.preprocessing_docs[topic]
        
        if self.console:
            content = f"[bold]{doc['description']}[/bold]\n\n"
            
            if 'methods' in doc:
                content += "[bold cyan]Methods:[/bold cyan]\n"
                for method, description in doc['methods'].items():
                    content += f"• [yellow]{method}:[/yellow] {description}\n"
            
            if 'considerations' in doc:
                content += "\n[bold red]Important Considerations:[/bold red]\n"
                for consideration in doc['considerations']:
                    content += f"• {consideration}\n"
            
            if 'when_to_use' in doc:
                content += "\n[bold green]When to Use:[/bold green]\n"
                for method, usage in doc['when_to_use'].items():
                    content += f"• [yellow]{method}:[/yellow] {usage}\n"
            
            self.console.print(Panel(content, title=f"📚 {topic} Guide", border_style="cyan"))
        else:
            print(f"\n{topic} Documentation")
            print("=" * 40)
            print(doc['description'])
    
    def show_metrics_explanation(self, problem_type: str):
        """Show detailed explanation of evaluation metrics."""
        metrics_type = "Classification Metrics" if problem_type == "classification" else "Regression Metrics"
        
        if metrics_type not in self.metrics_docs:
            return
        
        metrics = self.metrics_docs[metrics_type]
        
        if self.console:
            table = Table(title=f"📊 {metrics_type} Explanation", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", width=15)
            table.add_column("Formula", style="yellow", width=25)
            table.add_column("Interpretation", style="white", width=30)
            table.add_column("Best For", style="green", width=20)
            
            for metric_name, metric_info in metrics.items():
                table.add_row(
                    metric_name,
                    metric_info['formula'],
                    metric_info['interpretation'],
                    metric_info['best_for']
                )
            
            self.console.print(table)
        else:
            print(f"\n{metrics_type}")
            print("=" * 40)
            for metric_name, metric_info in metrics.items():
                print(f"{metric_name}: {metric_info['interpretation']}")
    
    def show_progress_with_steps(self, steps: List[str]):
        """Show progress indicator with step descriptions."""
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                for i, step in enumerate(steps):
                    task = progress.add_task(step, total=100)
                    time.sleep(0.1)  # Simulate work
                    progress.update(task, completed=100)
                    if i < len(steps) - 1:
                        progress.remove_task(task)
        else:
            for step in steps:
                print(f"Processing: {step}")
    
    def show_interactive_help(self):
        """Show comprehensive interactive help system."""
        while True:
            if self.console:
                self.console.print(Panel(
                    "[bold blue]📚 Interactive Help System[/bold blue]\n\n"
                    "Select a topic to learn more about:",
                    title="Help & Documentation",
                    border_style="blue"
                ))
            
            options = [
                "🤖 Machine Learning Models",
                "🔧 Preprocessing Techniques", 
                "📊 Evaluation Metrics",
                "💡 Best Practices",
                "🎯 Parameter Tuning Guide",
                "📈 Interpreting Results",
                "⬅️ Back to Main Menu"
            ]
            
            self.show_menu("Choose help topic:", options)
            choice = self.input("Enter your choice (1-7)", default="7")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self._show_models_help()
                elif choice == 2:
                    self._show_preprocessing_help_menu()
                elif choice == 3:
                    self._show_metrics_help_menu()
                elif choice == 4:
                    self._show_best_practices()
                elif choice == 5:
                    self._show_tuning_guide()
                elif choice == 6:
                    self._show_results_interpretation()
                elif choice == 7:
                    break
                else:
                    self.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.print("[red]❌ Please enter a valid number.[/red]")
            
            input("\nPress Enter to continue...")
    
    def _show_models_help(self):
        """Show help for machine learning models."""
        model_names = list(self.model_docs.keys())
        
        self.print("\n[bold]Available Models:[/bold]")
        for i, model in enumerate(model_names, 1):
            self.print(f"[cyan]{i}.[/cyan] {model}")
        
        try:
            choice = int(self.input(f"Select model (1-{len(model_names)}, 0 for overview)", default="0"))
            if choice == 0:
                self._show_model_overview()
            elif 1 <= choice <= len(model_names):
                self.show_model_documentation(model_names[choice-1])
        except ValueError:
            self.print("[red]❌ Invalid selection.[/red]")
    
    def _show_model_overview(self):
        """Show overview of all models."""
        if self.console:
            table = Table(title="🤖 Model Overview", box=box.ROUNDED)
            table.add_column("Model", style="cyan", width=20)
            table.add_column("Type", style="yellow", width=15)
            table.add_column("Best For", style="green", width=30)
            table.add_column("Complexity", style="red", width=15)
            
            model_types = {
                "KNN": ("Instance-based", "Small datasets, Non-linear patterns", "Low"),
                "SVM": ("Kernel-based", "High-dimensional data, Text classification", "Medium"),
                "Decision Tree": ("Tree-based", "Interpretable models, Mixed data types", "Low"),
                "Random Forest": ("Ensemble", "General purpose, Feature importance", "Medium"),
                "XGBoost": ("Gradient Boosting", "Competitions, Structured data", "High")
            }
            
            for model, (type_desc, best_for, complexity) in model_types.items():
                table.add_row(model, type_desc, best_for, complexity)
            
            self.console.print(table)
    
    def _show_best_practices(self):
        """Show ML best practices."""
        practices = [
            "🎯 Always establish a baseline model first (simple linear/tree model)",
            "📊 Understand your data before modeling (EDA is crucial)",
            "🔧 Preprocess data consistently across train/validation/test sets",
            "⚖️ Use appropriate evaluation metrics for your problem type",
            "🔄 Always use cross-validation for model selection",
            "📈 Monitor for overfitting (train vs validation performance)",
            "🎛️ Start with default parameters, then tune systematically",
            "📋 Keep detailed logs of experiments and results",
            "🧪 Test preprocessing steps on validation data",
            "🔍 Interpret model results and feature importance"
        ]
        
        if self.console:
            self.console.print(Panel(
                "\n".join(practices),
                title="💡 Machine Learning Best Practices",
                border_style="green"
            ))
        else:
            print("\nMachine Learning Best Practices:")
            for practice in practices:
                print(practice)

    def show_menu(self, title: str, options: List[str]):
        """Show enhanced menu with Rich formatting."""
        if self.console:
            self.console.print(f"\n[bold blue]{title}[/bold blue]")
            for i, option in enumerate(options, 1):
                self.console.print(f"[cyan]{i}.[/cyan] {option}")
        else:
            print(f"\n{title}")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")

class SmartDataAnalyzer:
    """
    Enhanced data analysis with comprehensive insights and recommendations.
    """
    
    def __init__(self, ui: EnhancedTerminalUI):
        self.ui = ui
        self.column_analysis = {}
        self.preprocessing_recommendations = {}
    
    def analyze_dataset_comprehensive(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        analysis = {
            'basic_info': self._get_basic_info(df),
            'data_quality': self._assess_data_quality(df),
            'feature_analysis': self._analyze_features(df, target_column),
            'target_analysis': self._analyze_target(df, target_column) if target_column else None,
            'correlations': self._analyze_correlations(df),
            'recommendations': self._generate_recommendations(df, target_column)
        }
        
        return analysis
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes_summary': df.dtypes.value_counts().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        
        quality_score = 100.0
        issues = []
        
        # Missing values assessment
        missing_percentage = (total_missing / df.size) * 100
        if missing_percentage > 20:
            quality_score -= 30
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            quality_score -= 15
            issues.append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        # Duplicate rows assessment
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 10:
            quality_score -= 20
            issues.append(f"High duplicate rows: {duplicate_percentage:.1f}%")
        
        # Feature variability assessment
        low_variance_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                low_variance_features.append(col)
                quality_score -= 5
        
        if low_variance_features:
            issues.append(f"Low variance features: {len(low_variance_features)}")
        
        return {
            'quality_score': max(0, quality_score),
            'total_missing': int(total_missing),
            'missing_percentage': missing_percentage,
            'issues': issues,
            'missing_by_column': missing_summary[missing_summary > 0].to_dict(),
            'recommendations': self._get_quality_recommendations(missing_percentage, duplicate_percentage, low_variance_features)
        }
    
    def _analyze_features(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze individual features."""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_column:
            if target_column in numerical_features:
                numerical_features.remove(target_column)
            if target_column in categorical_features:
                categorical_features.remove(target_column)
        
        analysis = {
            'numerical_features': {
                'count': len(numerical_features),
                'names': numerical_features,
                'summary': {},
                'outliers': {},
                'skewness': {}
            },
            'categorical_features': {
                'count': len(categorical_features),
                'names': categorical_features,
                'cardinality': {},
                'summary': {}
            }
        }
        
        # Analyze numerical features
        for col in numerical_features[:10]:  # Limit to first 10 for performance
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
                
                analysis['numerical_features']['summary'][col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median()
                }
                analysis['numerical_features']['outliers'][col] = len(outliers)
                analysis['numerical_features']['skewness'][col] = col_data.skew()
        
        # Analyze categorical features
        for col in categorical_features[:10]:  # Limit to first 10 for performance
            analysis['categorical_features']['cardinality'][col] = df[col].nunique()
            analysis['categorical_features']['summary'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_top3': df[col].value_counts().head(3).to_dict()
            }
        
        return analysis
    
    def _generate_recommendations(self, df: pd.DataFrame, target_column: str = None) -> List[str]:
        """Generate comprehensive preprocessing recommendations."""
        recommendations = []
        
        # Missing value recommendations
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            high_missing = missing_cols[missing_cols > len(df) * 0.5]
            if len(high_missing) > 0:
                recommendations.append(f"⚠️ Consider dropping columns with >50% missing: {list(high_missing.index)}")
            
            moderate_missing = missing_cols[(missing_cols > len(df) * 0.05) & (missing_cols <= len(df) * 0.5)]
            if len(moderate_missing) > 0:
                recommendations.append(f"🔧 Apply advanced imputation for: {list(moderate_missing.index)}")
        
        # Feature scaling recommendations
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            ranges = []
            for col in numerical_cols:
                if col != target_column and df[col].nunique() > 1:
                    col_range = df[col].max() - df[col].min()
                    ranges.append(col_range)
            
            if len(ranges) > 1 and max(ranges) / min(ranges) > 100:
                recommendations.append("📏 Features have very different scales - apply StandardScaler or RobustScaler")
        
        # High cardinality categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        for col in categorical_cols:
            if col != target_column and df[col].nunique() > 50:
                high_cardinality.append(col)
        
        if high_cardinality:
            recommendations.append(f"🏷️ High cardinality categorical features may need grouping: {high_cardinality}")
        
        # Outlier detection
        outlier_cols = []
        for col in numerical_cols:
            if col != target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    outlier_cols.append(col)
        
        if outlier_cols:
            recommendations.append(f"🎯 Consider outlier treatment for: {outlier_cols}")
        
        # Feature engineering suggestions
        if len(numerical_cols) >= 2:
            recommendations.append("⚙️ Consider creating interaction features between numerical variables")
        
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            recommendations.append("📅 Extract date features (day, month, year, weekday) if datetime columns exist")
        
        return recommendations


class EnhancedMLPipeline:
    """
    Enhanced ML Pipeline with comprehensive documentation, SQLite tracking, and improved UI.
    """
    
    def __init__(self, problem_type='auto', target_column=None, remove_outliers=True, 
                 test_size=0.2, cv_folds=5, random_state=RANDOM_STATE, 
                 experiment_name=None, user_notes=None):
        """
        Initialize Enhanced ML Pipeline.
        
        Args:
            problem_type: 'classification', 'regression', or 'auto' for automatic detection
            target_column: Name of target column
            remove_outliers: Whether to remove outliers from training data
            test_size: Proportion of data for validation (0.1 to 0.4)
            cv_folds: Number of cross-validation folds (3 to 10)
            random_state: Random seed for reproducibility
            experiment_name: Name for this experiment (for database tracking)
            user_notes: Additional notes about this experiment
        """
        # Initialize UI and database
        self.ui = EnhancedTerminalUI()
        self.db = ExperimentDatabase()
        self.analyzer = SmartDataAnalyzer(self.ui)
        
        # Configuration parameters
        self.problem_type = problem_type
        self.target_column = target_column
        self.remove_outliers = remove_outliers
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.experiment_name = experiment_name or f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_notes = user_notes
        
        # Data containers
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_submission = None
        
        # Preprocessing objects
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        
        # Model containers with detailed tracking
        self.available_models = {
            'KNN': True,
            'SVM': True,
            'Decision Tree': True,
            'Linear Model': True,
            'Random Forest': True,
            'Extra Trees': True,
            'Bagging': True,
            'AdaBoost': True,
            'Hist Gradient Boosting': True,
            'XGBoost': XGBOOST_AVAILABLE,
            'LightGBM': LIGHTGBM_AVAILABLE
        }
        self.models = {}
        self.model_metadata = {}  # Store training times, parameters, etc.
        self.predictions = {}
        self.scores = {}
        
        # Experiment tracking
        self.experiment_id = None
        self.experiment_start_time = None
        self.dataset_hash = None
        self.preprocessing_steps = []
        
        # Dataset info
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.is_binary_classification = False
        self.dataset_analysis = None
    
    def show_welcome(self):
        """Show enhanced welcome screen with comprehensive information."""
        self.ui.show_header(
            "🚀 Enhanced Machine Learning Pipeline v2.0",
            "With Comprehensive Documentation & SQLite Experiment Tracking"
        )
        
        if self.ui.console:
            welcome_panel = Panel(
                """[bold green]🎯 Welcome to the Enhanced ML Pipeline![/bold green]

[bold blue]🔥 New Features in v2.0:[/bold blue]
• 📚 Comprehensive parameter documentation with mathematical foundations
• 🗄️ SQLite database for experiment tracking and performance logging
• 🔍 Enhanced data analysis with smart preprocessing recommendations
• 📊 Interactive legends and tooltips for all functionality
• 🎛️ Advanced hyperparameter tuning with detailed explanations
• 📈 Improved visualizations and model comparison tools
• 💡 Best practices guide and parameter tuning assistance

[bold yellow]🚀 What You Can Do:[/bold yellow]
• 🔍 Explore your data with column-by-column analysis
• 📊 Get AI-powered preprocessing recommendations  
• 🤖 Train and compare 11 ML models with full documentation
• 📈 Visualize results with comprehensive model performance analysis
• 💾 Track all experiments in SQLite database for future reference
• 🎯 Generate predictions with confidence intervals

[bold cyan]📚 Built-in Help System:[/bold cyan]
• Interactive parameter explanations for all models
• Mathematical foundations and practical guidance
• Best practices for ML workflows
• Troubleshooting guides and optimization tips

[green]Let's build some amazing models! 🚀[/green]""",
                title="🏠 Enhanced ML Pipeline",
                border_style="blue",
                padding=(1, 2)
            )
            self.ui.console.print(welcome_panel)
        else:
            print("""
Enhanced Machine Learning Pipeline v2.0
=======================================

🔥 New Features:
• Comprehensive parameter documentation
• SQLite experiment tracking
• Enhanced data analysis
• Interactive help system
• Advanced visualizations

Ready to build amazing models!
            """)
    
    def main_menu(self):
        """Enhanced main menu with legends and help."""
        while True:
            self.ui.show_header("🏠 Main Menu", "Select an option or type 'help' for assistance")
            
            # Show legend for main menu
            self.ui.show_legend("main_menu")
            
            options = [
                "📁 Load Data & Configure Dataset",
                "🔍 Explore Data & Get Insights", 
                "🔧 Configure Preprocessing Pipeline",
                "🤖 Model Management & Documentation",
                "🚀 Run Complete ML Pipeline",
                "📊 View Results & Comparisons",
                "💾 Generate Submission File",
                "📚 Help & Documentation Center",
                "🗄️ View Experiment Database",
                "❌ Exit Pipeline"
            ]
            
            self.ui.show_menu("Select an option:", options)
            
            choice = self.ui.input("Enter your choice (1-10) or 'help' for assistance", default="1")
            
            # Handle help command
            if choice.lower() in ['help', 'h']:
                self.ui.show_interactive_help()
                continue
            
            try:
                choice = int(choice)
                if choice == 1:
                    self.load_data_menu()
                elif choice == 2:
                    self.explore_data_menu()
                elif choice == 3:
                    self.preprocessing_menu()
                elif choice == 4:
                    self.model_management_menu()
                elif choice == 5:
                    self.run_pipeline_menu()
                elif choice == 6:
                    self.view_results_menu()
                elif choice == 7:
                    self.generate_submission_menu()
                elif choice == 8:
                    self.ui.show_interactive_help()
                elif choice == 9:
                    self.view_database_menu()
                elif choice == 10:
                    self.ui.print("\n[bold yellow]👋 Thank you for using the Enhanced ML Pipeline![/bold yellow]")
                    self.ui.print("[green]🎯 Your experiments have been saved to the database for future reference.[/green]")
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice. Please try again or type 'help'.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number or 'help' for assistance.[/red]")
    
    def load_data_menu(self):
        """Enhanced data loading with comprehensive validation and analysis."""
        self.ui.show_header("📁 Data Loading & Configuration", "Load your datasets and configure the ML problem")
        
        # Show preprocessing legend
        self.ui.show_legend("preprocessing")
        
        train_file = self.ui.input("📂 Enter training data file path", default="train.csv")
        test_file = self.ui.input("📂 Enter test data file path (optional, press Enter to skip)", default="")
        target_col = self.ui.input("🎯 Enter target column name", default="target")
        
        # Get experiment metadata
        exp_name = self.ui.input("📝 Enter experiment name (optional)", default=self.experiment_name)
        user_notes = self.ui.input("📋 Enter experiment notes (optional)", default="")
        
        if exp_name:
            self.experiment_name = exp_name
        if user_notes:
            self.user_notes = user_notes
        
        if test_file.strip() == "":
            test_file = None
        
        # Start experiment timing
        self.experiment_start_time = time.time()
        
        success = self.load_data(train_file, test_file, target_col)
        
        if success:
            self.ui.print(f"[green]✅ Data loaded successfully![/green]")
            
            # Create experiment entry in database
            dataset_info = {
                'experiment_name': self.experiment_name,
                'dataset_name': Path(train_file).stem,
                'dataset_hash': self.dataset_hash,
                'problem_type': self.problem_type,
                'target_column': self.target_column,
                'n_samples': len(self.train_data),
                'n_features': len(self.train_data.columns) - 1,
                'train_size': 1 - self.test_size,
                'test_size': self.test_size,
                'cv_folds': self.cv_folds,
                'preprocessing_steps': [],
                'user_notes': self.user_notes
            }
            
            self.experiment_id = self.db.create_experiment(dataset_info)
            
            # Show comprehensive dataset overview
            self._show_dataset_overview()
            
            # Perform initial analysis
            if self.ui.confirm("🔍 Would you like to perform comprehensive dataset analysis now?"):
                self.dataset_analysis = self.analyzer.analyze_dataset_comprehensive(
                    self.train_data, self.target_column
                )
                self._display_dataset_analysis()
        
        input("\nPress Enter to continue...")
    
    def _show_dataset_overview(self):
        """Show comprehensive dataset overview."""
        if self.ui.console:
            # Basic info table
            basic_info = Table(title="📊 Dataset Overview", box=box.ROUNDED)
            basic_info.add_column("Attribute", style="cyan", width=20)
            basic_info.add_column("Value", style="white", width=30)
            basic_info.add_column("Description", style="green", width=40)
            
            info_data = [
                ("Problem Type", self.problem_type, "Automatically detected or user-specified"),
                ("Target Column", self.target_column, "Variable to predict"),
                ("Training Samples", f"{len(self.train_data):,}", "Number of rows in training data"),
                ("Features", f"{len(self.train_data.columns) - 1}", "Number of input variables"),
                ("Memory Usage", f"{self.train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "RAM required for dataset"),
                ("Missing Values", f"{self.train_data.isnull().sum().sum():,}", "Total missing data points"),
                ("Duplicate Rows", f"{self.train_data.duplicated().sum():,}", "Exact duplicate records")
            ]
            
            if self.test_data is not None:
                info_data.append(("Test Samples", f"{len(self.test_data):,}", "Number of rows in test data"))
            
            for attr, value, desc in info_data:
                basic_info.add_row(attr, value, desc)
            
            self.ui.console.print(basic_info)
            
            # Data types summary
            dtype_table = Table(title="📋 Data Types Summary", box=box.ROUNDED)
            dtype_table.add_column("Data Type", style="yellow")
            dtype_table.add_column("Count", style="cyan")
            dtype_table.add_column("Columns", style="white")
            
            for dtype, count in self.train_data.dtypes.value_counts().items():
                cols = [col for col in self.train_data.columns if self.train_data[col].dtype == dtype]
                dtype_table.add_row(str(dtype), str(count), ", ".join(cols[:5]) + ("..." if len(cols) > 5 else ""))
            
            self.ui.console.print(dtype_table)
        
        else:
            print(f"\nDataset Overview:")
            print(f"Problem Type: {self.problem_type}")
            print(f"Training samples: {len(self.train_data):,}")
            print(f"Features: {len(self.train_data.columns) - 1}")
            if self.test_data is not None:
                print(f"Test samples: {len(self.test_data):,}")
    
    def _display_dataset_analysis(self):
        """Display comprehensive dataset analysis results."""
        if not self.dataset_analysis:
            return
        
        analysis = self.dataset_analysis
        
        # Data quality assessment
        quality = analysis['data_quality']
        quality_color = "green" if quality['quality_score'] > 80 else "yellow" if quality['quality_score'] > 60 else "red"
        
        if self.ui.console:
            quality_panel = Panel(
                f"[bold]Overall Quality Score: [{quality_color}]{quality['quality_score']:.1f}/100[/{quality_color}][/bold]\n\n"
                f"📊 Missing Data: {quality['missing_percentage']:.1f}% ({quality['total_missing']:,} values)\n"
                f"🔄 Duplicate Rows: {analysis['basic_info']['duplicate_percentage']:.1f}%\n\n"
                f"[bold red]Issues Found:[/bold red]\n" + "\n".join([f"• {issue}" for issue in quality['issues']]) if quality['issues'] else "[green]No major issues detected![/green]",
                title="🔍 Data Quality Assessment",
                border_style=quality_color
            )
            self.ui.console.print(quality_panel)
            
            # Feature analysis summary
            feat_analysis = analysis['feature_analysis']
            feature_table = Table(title="📈 Feature Analysis Summary", box=box.ROUNDED)
            feature_table.add_column("Feature Type", style="cyan")
            feature_table.add_column("Count", style="yellow")
            feature_table.add_column("Key Insights", style="white")
            
            # Numerical features insights
            num_outliers = sum(feat_analysis['numerical_features']['outliers'].values())
            high_skew = len([k for k, v in feat_analysis['numerical_features']['skewness'].items() if abs(v) > 1])
            num_insights = f"Outliers in {len([k for k, v in feat_analysis['numerical_features']['outliers'].items() if v > 0])} features, {high_skew} highly skewed"
            
            feature_table.add_row(
                "Numerical", 
                str(feat_analysis['numerical_features']['count']),
                num_insights
            )
            
            # Categorical features insights
            high_card = len([k for k, v in feat_analysis['categorical_features']['cardinality'].items() if v > 50])
            cat_insights = f"{high_card} high-cardinality features" if high_card > 0 else "Normal cardinality levels"
            
            feature_table.add_row(
                "Categorical",
                str(feat_analysis['categorical_features']['count']),
                cat_insights
            )
            
            self.ui.console.print(feature_table)
            
            # Recommendations
            if analysis['recommendations']:
                rec_panel = Panel(
                    "\n".join([f"• {rec}" for rec in analysis['recommendations']]),
                    title="💡 Smart Preprocessing Recommendations",
                    border_style="blue"
                )
                self.ui.console.print(rec_panel)
        
        else:
            print(f"\nData Quality Score: {quality['quality_score']:.1f}/100")
            print(f"Missing Data: {quality['missing_percentage']:.1f}%")
            if quality['issues']:
                print("Issues:")
                for issue in quality['issues']:
                    print(f"  • {issue}")
    
    def explore_data_menu(self):
        """Enhanced data exploration with comprehensive analysis tools."""
        if self.train_data is None:
            self.ui.print("[red]❌ Please load data first![/red]")
            input("Press Enter to continue...")
            return
        
        while True:
            self.ui.show_header("🔍 Data Exploration & Analysis", "Comprehensive data insights and visualizations")
            
            options = [
                "📊 Dataset Overview & Quality Assessment",
                "🔍 Analyze Individual Column (with AI insights)",
                "📈 Correlation Analysis & Feature Relationships", 
                "🕳️ Missing Values Analysis & Patterns",
                "🎯 Target Variable Deep Dive",
                "🤖 Get AI-Powered Preprocessing Recommendations",
                "📉 Outlier Detection & Analysis",
                "📋 Generate Comprehensive Data Report",
                "⬅️ Back to Main Menu"
            ]
            
            self.ui.show_menu("Exploration Options:", options)
            choice = self.ui.input("Enter your choice (1-9)", default="1")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self.show_dataset_overview_detailed()
                elif choice == 2:
                    self.analyze_individual_column_enhanced()
                elif choice == 3:
                    self.show_correlation_analysis_enhanced()
                elif choice == 4:
                    self.show_missing_values_analysis_enhanced()
                elif choice == 5:
                    self.analyze_target_variable_enhanced()
                elif choice == 6:
                    self.get_ai_preprocessing_recommendations()
                elif choice == 7:
                    self.detect_and_analyze_outliers()
                elif choice == 8:
                    self.generate_comprehensive_report()
                elif choice == 9:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            input("\nPress Enter to continue...")
    
    def model_management_menu(self):
        """Enhanced model management with comprehensive documentation."""
        while True:
            self.ui.show_header("🤖 Model Management & Documentation", "Configure models and explore their capabilities")
            
            # Show model legend
            self.ui.show_legend("model_parameters")
            
            # Display current model status
            self._show_model_status_table()
            
            options = [
                "🔧 Enable/Disable Models for Training",
                "📚 View Model Documentation & Parameters",
                "⚙️ Configure Hyperparameter Search Spaces",
                "🎯 Get Model Recommendations for Your Data",
                "📊 Compare Model Characteristics",
                "🔍 Model Selection Wizard",
                "💡 View Best Practices for Model Selection",
                "⬅️ Back to Main Menu"
            ]
            
            self.ui.show_menu("Model Management Options:", options)
            choice = self.ui.input("Enter your choice (1-8)", default="1")
            
            try:
                choice = int(choice)
                if choice == 1:
                    self.toggle_models_enhanced()
                elif choice == 2:
                    self.view_model_documentation_menu()
                elif choice == 3:
                    self.configure_hyperparameter_spaces()
                elif choice == 4:
                    self.get_model_recommendations()
                elif choice == 5:
                    self.compare_model_characteristics()
                elif choice == 6:
                    self.model_selection_wizard()
                elif choice == 7:
                    self.show_model_selection_best_practices()
                elif choice == 8:
                    break
                else:
                    self.ui.print("[red]❌ Invalid choice.[/red]")
            except ValueError:
                self.ui.print("[red]❌ Please enter a valid number.[/red]")
            
            input("\nPress Enter to continue...")
    
    def _show_model_status_table(self):
        """Show current model status in a comprehensive table."""
        if self.ui.console:
            status_table = Table(title="🤖 Current Model Configuration", box=box.ROUNDED)
            status_table.add_column("Model", style="cyan", width=20)
            status_table.add_column("Status", style="white", width=15)
            status_table.add_column("Type", style="yellow", width=15)
            status_table.add_column("Best For", style="green", width=30)
            status_table.add_column("Complexity", style="red", width=12)
            
            model_info = {
                'KNN': ('Instance-based', 'Small datasets, non-linear patterns', 'Low'),
                'SVM': ('Kernel-based', 'High-dimensional data, text classification', 'Medium'),
                'Decision Tree': ('Tree-based', 'Interpretable models, mixed data', 'Low'),
                'Linear Model': ('Linear', 'Baseline models, linear relationships', 'Low'),
                'Random Forest': ('Ensemble', 'General purpose, robust performance', 'Medium'),
                'Extra Trees': ('Ensemble', 'Fast training, good generalization', 'Medium'),
                'Bagging': ('Ensemble', 'Reduce overfitting, stable predictions', 'Medium'),
                'AdaBoost': ('Boosting', 'Sequential improvement, weighted samples', 'Medium'),
                'Hist Gradient Boosting': ('Boosting', 'Large datasets, memory efficient', 'High'),
                'XGBoost': ('Boosting', 'Competitions, structured data', 'High'),
                'LightGBM': ('Boosting', 'Fast training, high accuracy', 'High')
            }
            
            for model_name, enabled in self.available_models.items():
                if enabled:
                    status = "✅ Enabled"
                    status_style = "green"
                else:
                    if model_name in ['XGBoost', 'LightGBM']:
                        if model_name == 'XGBoost' and not XGBOOST_AVAILABLE:
                            status = "❌ Not Available"
                            status_style = "red"
                        elif model_name == 'LightGBM' and not LIGHTGBM_AVAILABLE:
                            status = "❌ Not Available"
                            status_style = "red"
                        else:
                            status = "⏸️ Disabled"
                            status_style = "yellow"
                    else:
                        status = "⏸️ Disabled"
                        status_style = "yellow"
                
                info = model_info.get(model_name, ('Unknown', 'General purpose', 'Medium'))
                
                status_table.add_row(
                    model_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    info[0],
                    info[1],
                    info[2]
                )
            
            self.ui.console.print(status_table)
        else:
            print("\nCurrent Model Status:")
            for model_name, enabled in self.available_models.items():
                status = "Enabled" if enabled else "Disabled"
                print(f"  {model_name}: {status}")
    
    def view_model_documentation_menu(self):
        """Enhanced model documentation viewer."""
        model_names = list(self.ui.model_docs.keys())
        
        self.ui.print("\n[bold]📚 Available Model Documentation:[/bold]")
        for i, model in enumerate(model_names, 1):
            availability = ""
            if model == 'XGBoost' and not XGBOOST_AVAILABLE:
                availability = " [red](Not Available)[/red]"
            elif model == 'LightGBM' and not LIGHTGBM_AVAILABLE:
                availability = " [red](Not Available)[/red]"
            
            self.ui.print(f"[cyan]{i}.[/cyan] {model}{availability}")
        
        try:
            choice = int(self.ui.input(f"Select model for documentation (1-{len(model_names)}, 0 for comparison)", default="0"))
            if choice == 0:
                self.compare_model_characteristics()
            elif 1 <= choice <= len(model_names):
                model_name = model_names[choice-1]
                self.ui.show_model_documentation(model_name)
                
                # Offer parameter tuning tips
                if self.ui.confirm(f"\n🎯 Would you like parameter tuning tips for {model_name}?"):
                    self._show_parameter_tuning_tips(model_name)
        except ValueError:
            self.ui.print("[red]❌ Invalid selection.[/red]")
    
    def _show_parameter_tuning_tips(self, model_name: str):
        """Show specific parameter tuning tips for a model."""
        tips = {
            'KNN': [
                "🎯 Start with k = sqrt(n_samples) and adjust based on validation performance",
                "📏 Always use feature scaling - KNN is distance-based",
                "🔍 Try both 'uniform' and 'distance' weights",
                "📊 Use odd k values for binary classification to avoid ties",
                "⚡ Consider using approximate nearest neighbors for large datasets"
            ],
            'SVM': [
                "🎯 Start with C=1.0 and RBF kernel for most problems", 
                "📏 Feature scaling is mandatory - use StandardScaler",
                "🔍 Try linear kernel first if you have many features (>10k)",
                "⚙️ Use gamma='scale' initially, then fine-tune",
                "📊 For large datasets, consider using linear SVM or SGD"
            ],
            'Random Forest': [
                "🌳 Start with n_estimators=100, increase until performance plateaus",
                "📊 Use max_features='sqrt' for classification, 1/3 for regression",
                "🎯 Let max_depth=None initially, reduce only if overfitting",
                "⚡ Increase min_samples_split for noisy data",
                "📈 Use out-of-bag score for quick performance estimation"
            ],
            'XGBoost': [
                "🎯 Start with learning_rate=0.1 and n_estimators=100",
                "📊 Use early stopping to find optimal n_estimators",
                "🔧 Try max_depth between 3-8, start with 6",
                "⚖️ Lower learning_rate with more estimators often works better",
                "📈 Use subsample=0.8-0.9 to prevent overfitting"
            ]
        }
        
        if model_name in tips:
            if self.ui.console:
                tip_panel = Panel(
                    "\n".join(tips[model_name]),
                    title=f"💡 {model_name} Parameter Tuning Tips",
                    border_style="yellow"
                )
                self.ui.console.print(tip_panel)
            else:
                print(f"\n{model_name} Tuning Tips:")
                for tip in tips[model_name]:
                    print(f"  {tip}")
    
    def load_data(self, train_path: str, test_path: str = None, target_column: str = None) -> bool:
        """
        Enhanced data loading with validation and hash generation.
        
        Args:
            train_path: Path to training data CSV file
            test_path: Path to test data CSV file (optional)
            target_column: Name of target column
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Load training data
            self.train_data = pd.read_csv(train_path)
            self.ui.print(f"[green]✅ Training data loaded: {self.train_data.shape}[/green]")
            
            # Generate dataset hash for tracking
            self.dataset_hash = hashlib.md5(
                str(self.train_data.values.tobytes()).encode()
            ).hexdigest()[:16]
            
            # Load test data if provided
            if test_path:
                self.test_data = pd.read_csv(test_path)
                self.ui.print(f"[green]✅ Test data loaded: {self.test_data.shape}[/green]")
            
            # Set target column
            if target_column:
                self.target_column = target_column
            
            # Validate target column exists
            if self.target_column not in self.train_data.columns:
                self.ui.print(f"[red]❌ Target column '{self.target_column}' not found in training data![/red]")
                self.ui.print(f"Available columns: {list(self.train_data.columns)}")
                return False
            
            # Auto-detect problem type if not specified
            if self.problem_type == 'auto':
                self._detect_problem_type()
                self.ui.print(f"[blue]🔍 Auto-detected problem type: {self.problem_type}[/blue]")
            
            # Identify feature types
            self._identify_feature_types()
            
            return True
            
        except FileNotFoundError as e:
            self.ui.print(f"[red]❌ File not found: {e}[/red]")
            return False
        except pd.errors.EmptyDataError:
            self.ui.print(f"[red]❌ Empty CSV file provided[/red]")
            return False
        except Exception as e:
            self.ui.print(f"[red]❌ Error loading data: {str(e)}[/red]")
            return False
        
    # Training methods for individual models
    def train_random_forest_model(self):
        """Train Random Forest with comprehensive parameter tuning."""
        print(f"🔄 Training Random Forest {'Classifier' if self.problem_type == 'classification' else 'Regressor'}...")
        
        if self.problem_type == 'classification':
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
        else:
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        grid_search = GridSearchCV(
            model, params,
            cv=min(3, self.cv_folds),
            scoring=self._get_scoring_metric(),
            n_jobs=1,  # Model already uses n_jobs
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best Random Forest params: {grid_search.best_params_}")
        print(f"   ✅ Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

    def train_extra_trees_model(self):
        """Train Extra Trees with comprehensive parameter tuning.""" 
        print(f"🔄 Training Extra Trees {'Classifier' if self.problem_type == 'classification' else 'Regressor'}...")
        
        if self.problem_type == 'classification':
            model = ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            model = ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1)
        
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            model, params,
            cv=3,
            scoring=self._get_scoring_metric(),
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best Extra Trees params: {grid_search.best_params_}")
        print(f"   ✅ Best Extra Trees CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

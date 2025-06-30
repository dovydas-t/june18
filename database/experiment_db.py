"""
Enhanced experiment database with improved tracking and querying capabilities
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ExperimentDatabase:
    """
    Enhanced experiment tracking database with comprehensive functionality.
    """
    
    def __init__(self, db_path: str = "ml_experiments.db"):
        """Initialize database with comprehensive schema."""
        self.db_path = db_path
        self.connection = None
        try:
            self._initialize_database()
        except Exception as e:
            print(f"Warning: Database initialization failed: {e}")
    
    def _initialize_database(self):
        """Initialize database with comprehensive tables."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.connection.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                dataset_name TEXT,
                dataset_hash TEXT,
                dataset_path TEXT,
                problem_type TEXT,
                target_column TEXT,
                n_samples INTEGER,
                n_features INTEGER,
                train_size REAL,
                test_size REAL,
                cv_folds INTEGER,
                preprocessing_steps TEXT,
                user_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                model_name TEXT NOT NULL,
                model_type TEXT,
                hyperparameters TEXT,
                cv_scores TEXT,
                mean_cv_score REAL,
                std_cv_score REAL,
                training_time REAL,
                prediction_time REAL,
                feature_importance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # Metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_result_id INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metric_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_result_id) REFERENCES model_results (id)
            )
        ''')
        
        # Dataset registry table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                dataset_hash TEXT,
                file_path TEXT,
                problem_type TEXT,
                target_column TEXT,
                n_samples INTEGER,
                n_features INTEGER,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
    
    def create_experiment(self, experiment_info: Dict[str, Any]) -> Optional[int]:
        """Create new experiment entry with comprehensive information."""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Register dataset if not exists
            self._register_dataset(experiment_info)
            
            cursor.execute('''
                INSERT INTO experiments (
                    experiment_name, dataset_name, dataset_hash, problem_type,
                    target_column, n_samples, n_features, train_size, test_size,
                    cv_folds, preprocessing_steps, user_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_info.get('experiment_name', 'Unknown'),
                experiment_info.get('dataset_name', 'Unknown'),
                experiment_info.get('dataset_hash', ''),
                experiment_info.get('problem_type', 'Unknown'),
                experiment_info.get('target_column', ''),
                experiment_info.get('n_samples', 0),
                experiment_info.get('n_features', 0),
                experiment_info.get('train_size', 0.8),
                experiment_info.get('test_size', 0.2),
                experiment_info.get('cv_folds', 5),
                json.dumps(experiment_info.get('preprocessing_steps', [])),
                experiment_info.get('user_notes', '')
            ))
            
            experiment_id = cursor.lastrowid
            self.connection.commit()
            return experiment_id
        except Exception as e:
            print(f"Error creating experiment: {e}")
            return None
    
    def _register_dataset(self, experiment_info: Dict[str, Any]):
        """Register dataset in the registry."""
        try:
            cursor = self.connection.cursor()
            
            # Check if dataset already exists
            cursor.execute(
                'SELECT id FROM dataset_registry WHERE dataset_name = ?',
                (experiment_info.get('dataset_name', ''),)
            )
            
            if cursor.fetchone() is None:
                # Insert new dataset
                cursor.execute('''
                    INSERT INTO dataset_registry (
                        dataset_name, dataset_hash, problem_type, target_column,
                        n_samples, n_features
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    experiment_info.get('dataset_name', ''),
                    experiment_info.get('dataset_hash', ''),
                    experiment_info.get('problem_type', ''),
                    experiment_info.get('target_column', ''),
                    experiment_info.get('n_samples', 0),
                    experiment_info.get('n_features', 0)
                ))
            else:
                # Update last_used timestamp
                cursor.execute(
                    'UPDATE dataset_registry SET last_used = CURRENT_TIMESTAMP WHERE dataset_name = ?',
                    (experiment_info.get('dataset_name', ''),)
                )
            
            self.connection.commit()
        except Exception as e:
            print(f"Error registering dataset: {e}")
    
    def save_model_results(self, experiment_id: int, model_name: str, results: Dict[str, Any]) -> Optional[int]:
        """Save model results to database."""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Insert model result
            cursor.execute('''
                INSERT INTO model_results (
                    experiment_id, model_name, model_type, hyperparameters,
                    cv_scores, mean_cv_score, std_cv_score, training_time,
                    feature_importance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                model_name,
                results.get('model_type', 'Unknown'),
                json.dumps(results.get('hyperparameters', {})),
                json.dumps(results.get('cv_scores', [])),
                results.get('mean_cv_score', 0.0),
                results.get('std_cv_score', 0.0),
                results.get('training_time', 0.0),
                json.dumps(results.get('feature_importance', {}))
            ))
            
            model_result_id = cursor.lastrowid
            
            # Insert individual metrics
            metrics = results.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO metrics (model_result_id, metric_name, metric_value, metric_type)
                    VALUES (?, ?, ?, ?)
                ''', (
                    model_result_id,
                    metric_name,
                    float(metric_value) if isinstance(metric_value, (int, float)) else 0.0,
                    'performance'
                ))
            
            self.connection.commit()
            return model_result_id
            
        except Exception as e:
            print(f"Error saving model results: {e}")
            return None
    
    def get_experiment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent experiments with comprehensive information."""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT 
                    id, experiment_name, dataset_name, dataset_hash, problem_type,
                    target_column, n_samples, n_features, created_at
                FROM experiments 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'experiment_name': row[1],
                    'dataset_name': row[2],
                    'dataset_hash': row[3],
                    'problem_type': row[4],
                    'target_column': row[5],
                    'n_samples': row[6],
                    'n_features': row[7],
                    'created_at': row[8]
                })
            return results
        except Exception as e:
            print(f"Error getting experiment history: {e}")
            return []
    
    def get_dataset_registry(self) -> List[Dict[str, Any]]:
        """Get all registered datasets."""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT 
                    dataset_name, problem_type, target_column, n_samples,
                    n_features, last_used, created_at
                FROM dataset_registry 
                ORDER BY last_used DESC
            ''')
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'dataset_name': row[0],
                    'problem_type': row[1],
                    'target_column': row[2],
                    'n_samples': row[3],
                    'n_features': row[4],
                    'last_used': row[5],
                    'created_at': row[6]
                })
            return results
        except Exception as e:
            print(f"Error getting dataset registry: {e}")
            return []
    
    def get_best_models(self, limit: int = 10, metric_name: str = None) -> List[Dict[str, Any]]:
        """Get best performing models across all experiments."""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            if metric_name:
                # Filter by specific metric
                cursor.execute('''
                    SELECT 
                        mr.model_name, e.experiment_name, e.problem_type,
                        mr.mean_cv_score, mr.training_time, m.metric_value,
                        mr.created_at
                    FROM model_results mr
                    JOIN experiments e ON mr.experiment_id = e.id
                    JOIN metrics m ON mr.id = m.model_result_id
                    WHERE m.metric_name = ?
                    ORDER BY m.metric_value DESC
                    LIMIT ?
                ''', (metric_name, limit))
            else:
                # Get all models ordered by CV score
                cursor.execute('''
                    SELECT 
                        mr.model_name, e.experiment_name, e.problem_type,
                        mr.mean_cv_score, mr.training_time, mr.created_at
                    FROM model_results mr
                    JOIN experiments e ON mr.experiment_id = e.id
                    ORDER BY mr.mean_cv_score DESC
                    LIMIT ?
                ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                result_dict = {
                    'model_name': row[0],
                    'experiment_name': row[1],
                    'problem_type': row[2],
                    'mean_cv_score': row[3],
                    'training_time': row[4],
                    'created_at': row[-1]
                }
                if metric_name:
                    result_dict['metric_value'] = row[5]
                results.append(result_dict)
            
            return results
        except Exception as e:
            print(f"Error getting best models: {e}")
            return []
    
    def get_model_comparison(self, experiment_id: int = None) -> pd.DataFrame:
        """Get model comparison data as DataFrame."""
        if not self.connection:
            return pd.DataFrame()
        
        try:
            if experiment_id:
                query = '''
                    SELECT 
                        mr.model_name,
                        m.metric_name,
                        m.metric_value,
                        mr.training_time,
                        mr.mean_cv_score
                    FROM model_results mr
                    JOIN metrics m ON mr.id = m.model_result_id
                    WHERE mr.experiment_id = ?
                    ORDER BY mr.model_name, m.metric_name
                '''
                df = pd.read_sql_query(query, self.connection, params=(experiment_id,))
            else:
                query = '''
                    SELECT 
                        mr.model_name,
                        e.experiment_name,
                        m.metric_name,
                        m.metric_value,
                        mr.training_time,
                        mr.mean_cv_score
                    FROM model_results mr
                    JOIN experiments e ON mr.experiment_id = e.id
                    JOIN metrics m ON mr.id = m.model_result_id
                    ORDER BY e.created_at DESC, mr.model_name, m.metric_name
                '''
                df = pd.read_sql_query(query, self.connection)
            
            return df
        except Exception as e:
            print(f"Error getting model comparison: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get comprehensive database statistics."""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Count experiments
            cursor.execute('SELECT COUNT(*) FROM experiments')
            total_experiments = cursor.fetchone()[0]
            
            # Count model results
            cursor.execute('SELECT COUNT(*) FROM model_results')
            total_model_results = cursor.fetchone()[0]
            
            # Count metrics
            cursor.execute('SELECT COUNT(*) FROM metrics')
            total_metrics = cursor.fetchone()[0]
            
            # Count datasets
            cursor.execute('SELECT COUNT(*) FROM dataset_registry')
            total_datasets = cursor.fetchone()[0]
            
            return {
                'total_experiments': total_experiments,
                'total_model_results': total_model_results,
                'total_metrics': total_metrics,
                'total_datasets': total_datasets
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def export_results(self, filename: str = "experiment_results.csv") -> bool:
        """Export all results to CSV file."""
        if not self.connection:
            return False
        
        try:
            query = '''
                SELECT 
                    e.experiment_name,
                    e.dataset_name,
                    e.problem_type,
                    e.target_column,
                    e.n_samples,
                    e.n_features,
                    mr.model_name,
                    m.metric_name,
                    m.metric_value,
                    mr.training_time,
                    mr.mean_cv_score,
                    mr.std_cv_score,
                    e.created_at
                FROM experiments e
                JOIN model_results mr ON e.id = mr.experiment_id
                JOIN metrics m ON mr.id = m.model_result_id
                ORDER BY e.created_at DESC, mr.model_name, m.metric_name
            '''
            
            df = pd.read_sql_query(query, self.connection)
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
    
    def search_experiments(self, dataset_name: str = None, problem_type: str = None, 
                          model_name: str = None) -> List[Dict[str, Any]]:
        """Search experiments with filters."""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            query = '''
                SELECT DISTINCT
                    e.id, e.experiment_name, e.dataset_name, e.problem_type,
                    e.created_at
                FROM experiments e
                LEFT JOIN model_results mr ON e.id = mr.experiment_id
                WHERE 1=1
            '''
            params = []
            
            if dataset_name:
                query += ' AND e.dataset_name LIKE ?'
                params.append(f'%{dataset_name}%')
            
            if problem_type:
                query += ' AND e.problem_type = ?'
                params.append(problem_type)
            
            if model_name:
                query += ' AND mr.model_name = ?'
                params.append(model_name)
            
            query += ' ORDER BY e.created_at DESC'
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'experiment_name': row[1],
                    'dataset_name': row[2],
                    'problem_type': row[3],
                    'created_at': row[4]
                })
            
            return results
        except Exception as e:
            print(f"Error searching experiments: {e}")
            return []
    
    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment and all related data."""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Delete metrics first (foreign key constraint)
            cursor.execute('''
                DELETE FROM metrics 
                WHERE model_result_id IN (
                    SELECT id FROM model_results WHERE experiment_id = ?
                )
            ''', (experiment_id,))
            
            # Delete model results
            cursor.execute('DELETE FROM model_results WHERE experiment_id = ?', (experiment_id,))
            
            # Delete experiment
            cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error deleting experiment: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
"""
Enhanced data exploration engine with comprehensive analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ExplorationEngine:
    """Comprehensive data exploration and analysis engine with smart recommendations."""
    
    def __init__(self, ui, analyzer=None):
        self.ui = ui
        self.analyzer = analyzer
        self.recommendations_cache = {}
    
    def show_interactive_dashboard(self, df: pd.DataFrame, target_column: str = None, problem_type: str = None):
        """Show interactive data exploration dashboard."""
        self.ui.print("\n📊 INTERACTIVE DATA DASHBOARD v3.1")
        self.ui.print("="*45)
        
        # Basic dataset overview
        self.ui.print(f"📋 Dataset Overview:")
        self.ui.print(f"   Shape: {df.shape}")
        self.ui.print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if target_column:
            self.ui.print(f"   Target: {target_column} ({problem_type})")
        
        # Data types summary
        dtype_summary = df.dtypes.value_counts()
        self.ui.print(f"   Data types: {dict(dtype_summary)}")
        
        # Missing values summary
        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / df.size) * 100
        self.ui.print(f"   Missing values: {missing_total:,} ({missing_pct:.1f}%)")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        self.ui.print(f"   Duplicate rows: {duplicates:,}")
    
    def show_statistical_analysis_suite(self, df: pd.DataFrame, target_column: str = None):
        """Show comprehensive statistical analysis."""
        self.ui.print("\n📈 STATISTICAL ANALYSIS SUITE v3.1")
        self.ui.print("="*45)
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        # Numerical features analysis
        if numerical_cols:
            self.ui.print(f"📊 Numerical Features Analysis ({len(numerical_cols)} features):")
            
            for col in numerical_cols[:5]:  # Show first 5
                stats = df[col].describe()
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()
                
                self.ui.print(f"\n   🔢 {col}:")
                self.ui.print(f"      Range: {stats['min']:.2f} to {stats['max']:.2f}")
                self.ui.print(f"      Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                self.ui.print(f"      Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
                
                if abs(skewness) > 1:
                    direction = "right" if skewness > 0 else "left"
                    self.ui.print(f"      💡 {direction}-skewed distribution")
        
        # Categorical features analysis
        if categorical_cols:
            self.ui.print(f"\n🏷️ Categorical Features Analysis ({len(categorical_cols)} features):")
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                top_category = df[col].mode()[0] if not df[col].empty else "N/A"
                top_freq = (df[col] == top_category).sum() if top_category != "N/A" else 0
                top_pct = (top_freq / len(df)) * 100
                
                self.ui.print(f"\n   📋 {col}:")
                self.ui.print(f"      Unique values: {unique_count}")
                self.ui.print(f"      Most common: '{top_category}' ({top_pct:.1f}%)")
                
                if unique_count > 50:
                    self.ui.print(f"      ⚠️ High cardinality - consider grouping")
                elif unique_count == 1:
                    self.ui.print(f"      ⚠️ Constant column - consider removing")
    
    def show_data_quality_assessment(self, df: pd.DataFrame, dataset_analysis: Dict = None):
        """Show comprehensive data quality assessment."""
        self.ui.print("\n🔍 DATA QUALITY ASSESSMENT v3.1")
        self.ui.print("="*45)
        
        # Calculate quality metrics
        quality_score = 100.0
        issues = []
        recommendations = []
        
        # Missing values assessment
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        missing_pct = (missing_summary.sum() / df.size) * 100
        
        if missing_pct > 20:
            quality_score -= 30
            issues.append(f"High missing data: {missing_pct:.1f}%")
            recommendations.append("Address missing values with imputation or removal")
        elif missing_pct > 5:
            quality_score -= 15
            issues.append(f"Moderate missing data: {missing_pct:.1f}%")
        
        # Duplicate assessment
        duplicates = df.duplicated().sum()
        duplicate_pct = (duplicates / len(df)) * 100
        
        if duplicate_pct > 10:
            quality_score -= 20
            issues.append(f"High duplicate rows: {duplicate_pct:.1f}%")
            recommendations.append("Remove duplicate rows")
        elif duplicate_pct > 0:
            issues.append(f"Some duplicate rows: {duplicate_pct:.1f}%")
        
        # Constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_score -= 10
            issues.append(f"Constant columns found: {len(constant_cols)}")
            recommendations.append("Remove constant columns")
        
        # High cardinality categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        high_card_cols = []
        for col in categorical_cols:
            if df[col].nunique() > 50:
                high_card_cols.append(col)
        
        if high_card_cols:
            quality_score -= 10
            issues.append(f"High cardinality categoricals: {len(high_card_cols)}")
            recommendations.append("Group rare categories or use target encoding")
        
        # Display results
        quality_score = max(0, quality_score)
        
        if quality_score >= 90:
            quality_label = "Excellent ✅"
        elif quality_score >= 75:
            quality_label = "Good 👍"
        elif quality_score >= 60:
            quality_label = "Fair ⚠️"
        else:
            quality_label = "Needs Attention ❌"
        
        self.ui.print(f"📊 Overall Quality Score: {quality_score:.1f}/100 ({quality_label})")
        
        if issues:
            self.ui.print(f"\n❌ Issues Found:")
            for issue in issues:
                self.ui.print(f"   • {issue}")
        
        if recommendations:
            self.ui.print(f"\n💡 Recommendations:")
            for rec in recommendations:
                self.ui.print(f"   • {rec}")
        
        if not issues:
            self.ui.print("\n✅ No major data quality issues detected!")
    
    def show_feature_relationships_analysis(self, df: pd.DataFrame, target_column: str = None):
        """Analyze relationships between features and target."""
        self.ui.print("\n🔗 FEATURE RELATIONSHIPS ANALYSIS v3.1")
        self.ui.print("="*50)
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if len(numerical_cols) < 2:
            self.ui.print("❌ Need at least 2 numerical columns for correlation analysis")
            return
        
        # Correlation with target
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                self.ui.print(f"🎯 Correlations with Target ({target_column}):")
                
                correlations = df[numerical_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
                
                for col, corr in correlations.head(10).items():
                    strength = self._get_correlation_strength(corr)
                    self.ui.print(f"   {col}: {corr:.3f} ({strength})")
        
        # Feature correlations
        self.ui.print(f"\n🔗 Inter-feature Correlations:")
        corr_matrix = df[numerical_cols].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            self.ui.print("   ⚠️ Highly correlated pairs (>0.8):")
            for col1, col2, corr in high_corr_pairs:
                self.ui.print(f"      {col1} ↔ {col2}: {corr:.3f}")
        else:
            self.ui.print("   ✅ No highly correlated feature pairs found")
    
    def _get_correlation_strength(self, corr: float) -> str:
        """Get correlation strength description."""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "Strong"
        elif abs_corr > 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    def show_smart_data_insights(self, df: pd.DataFrame, target_column: str = None):
        """Show AI-powered smart insights and recommendations."""
        self.ui.print("\n🧠 SMART DATA INSIGHTS & RECOMMENDATIONS v3.1")
        self.ui.print("="*60)
        
        insights = self._generate_smart_insights(df, target_column)
        
        # Data Quality Insights
        self.ui.print("🔍 Data Quality Insights:")
        for insight in insights['quality']:
            self.ui.print(f"   {insight}")
        
        # Feature Engineering Recommendations
        if insights['feature_engineering']:
            self.ui.print("\n🔧 Feature Engineering Recommendations:")
            for rec in insights['feature_engineering']:
                self.ui.print(f"   {rec}")
        
        # Distribution Insights
        if insights['distributions']:
            self.ui.print("\n📊 Distribution Analysis:")
            for dist in insights['distributions']:
                self.ui.print(f"   {dist}")
        
        # Modeling Recommendations
        if insights['modeling']:
            self.ui.print("\n🤖 Modeling Recommendations:")
            for model in insights['modeling']:
                self.ui.print(f"   {model}")
        
        # Data Issues & Fixes
        if insights['issues']:
            self.ui.print("\n⚠️ Potential Issues & Fixes:")
            for issue in insights['issues']:
                self.ui.print(f"   {issue}")
    
    def _generate_smart_insights(self, df: pd.DataFrame, target_column: str = None) -> Dict:
        """Generate comprehensive smart insights."""
        insights = {
            'quality': [],
            'feature_engineering': [],
            'distributions': [],
            'modeling': [],
            'issues': []
        }
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_column:
            if target_column in numerical_cols:
                numerical_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        # Data Quality Analysis
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct < 5:
            insights['quality'].append("✅ Excellent data quality - minimal missing values")
        elif missing_pct < 15:
            insights['quality'].append("⚠️ Good data quality - some missing values to handle")
        else:
            insights['quality'].append("❌ Data quality needs attention - significant missing values")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            insights['quality'].append(f"🔄 {duplicates} duplicate rows detected - consider removing")
        
        # Feature Engineering Opportunities
        if len(numerical_cols) >= 2:
            # Check for ratio opportunities
            high_corr_pairs = self._find_correlated_pairs(df, numerical_cols)
            if high_corr_pairs:
                insights['feature_engineering'].append(f"🔗 {len(high_corr_pairs)} highly correlated pairs found - consider creating ratios or differences")
        
        # Distribution Analysis
        for col in numerical_cols[:5]:  # Analyze first 5 numerical columns
            skewness = df[col].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                transform = "log" if skewness > 0 else "power"
                insights['distributions'].append(f"📈 {col} is {direction}-skewed → Consider {transform} transformation")
        
        # Categorical Analysis
        for col in categorical_cols:
            cardinality = df[col].nunique()
            if cardinality > 50:
                insights['issues'].append(f"🏷️ {col} has high cardinality ({cardinality}) → Group rare categories")
            elif cardinality == 1:
                insights['issues'].append(f"🏷️ {col} is constant → Remove from modeling")
        
        # Modeling Recommendations
        if target_column:
            target_type = "classification" if target_column in categorical_cols or df[target_column].nunique() < 10 else "regression"
            
            if target_type == "classification":
                class_balance = df[target_column].value_counts()
                min_class_pct = (class_balance.min() / len(df)) * 100
                
                if min_class_pct < 5:
                    insights['modeling'].append("⚖️ Imbalanced classes detected → Consider SMOTE or class weighting")
                
                if df[target_column].nunique() == 2:
                    insights['modeling'].append("🎯 Binary classification → Random Forest, XGBoost, or Logistic Regression recommended")
                else:
                    insights['modeling'].append("🎯 Multi-class classification → Random Forest or XGBoost recommended")
            else:
                target_skew = df[target_column].skew()
                if abs(target_skew) > 1:
                    insights['modeling'].append("📊 Skewed target → Consider log transformation for better model performance")
                
                insights['modeling'].append("📈 Regression problem → Random Forest, XGBoost, or Linear models recommended")
        
        # Feature Selection Insights
        if len(numerical_cols) > 20:
            insights['modeling'].append(f"📊 High dimensional data ({len(numerical_cols)} features) → Consider feature selection")
        
        return insights
    
    def _find_correlated_pairs(self, df, numerical_cols):
        """Find highly correlated column pairs."""
        corr_matrix = df[numerical_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        return high_corr_pairs
    
    def show_distribution_fixer(self, df: pd.DataFrame, target_column: str = None):
        """Interactive distribution analysis and fixing tool."""
        self.ui.print("\n📊 SMART DISTRIBUTION FIXER v3.1")
        self.ui.print("="*45)
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if len(numerical_cols) == 0:
            self.ui.print("❌ No numerical columns to analyze")
            return
        
        self.ui.print(f"Analyzing {len(numerical_cols)} numerical columns...")
        
        for col in numerical_cols:
            self._analyze_and_fix_distribution(df, col)
    
    def _analyze_and_fix_distribution(self, df, col):
        """Analyze and provide fixing options for a single column."""
        self.ui.print(f"\n🔍 Analyzing: {col}")
        
        # Basic statistics
        stats = df[col].describe()
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        
        self.ui.print(f"   Range: {stats['min']:.2f} to {stats['max']:.2f}")
        self.ui.print(f"   Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
        self.ui.print(f"   Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
        
        # Identify issues and provide recommendations
        issues = []
        fixes = []
        
        if abs(skewness) > 1:
            if skewness > 1:
                issues.append("Right-skewed distribution")
                if (df[col] > 0).all():
                    fixes.append("Log transformation")
                else:
                    fixes.append("Add constant then log transformation")
                fixes.append("Square root transformation")
            else:
                issues.append("Left-skewed distribution")
                fixes.append("Square transformation")
                fixes.append("Exponential transformation")
        
        if kurtosis > 3:
            issues.append("Heavy-tailed distribution (high kurtosis)")
            fixes.append("Winsorization (cap extreme values)")
        
        # Outlier detection
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(df)) * 100
            issues.append(f"{outliers} outliers ({outlier_pct:.1f}%)")
            fixes.append("Remove outliers")
            fixes.append("Cap outliers at percentiles")
        
        # Zero/negative values
        if (df[col] <= 0).any():
            zero_count = (df[col] <= 0).sum()
            issues.append(f"{zero_count} zero/negative values")
            fixes.append("Add constant before log transformation")
        
        # Display results
        if issues:
            self.ui.print(f"   ⚠️ Issues: {', '.join(issues)}")
            self.ui.print(f"   💡 Suggested fixes: {', '.join(fixes)}")
            
            # Offer to apply fix
            if len(fixes) > 0:
                apply_fix = self.ui.confirm(f"Apply recommended fix for {col}?", default=False)
                if apply_fix:
                    self._apply_distribution_fix(df, col, fixes[0])
        else:
            self.ui.print("   ✅ Distribution looks good!")
    
    def _apply_distribution_fix(self, df, col, fix_type):
        """Apply the selected distribution fix."""
        try:
            if fix_type == "Log transformation":
                df[f"{col}_log"] = np.log(df[col])
                self.ui.print(f"   ✅ Created {col}_log with log transformation")
            
            elif fix_type == "Square root transformation":
                df[f"{col}_sqrt"] = np.sqrt(df[col])
                self.ui.print(f"   ✅ Created {col}_sqrt with square root transformation")
            
            elif fix_type == "Add constant then log transformation":
                constant = abs(df[col].min()) + 1
                df[f"{col}_log"] = np.log(df[col] + constant)
                self.ui.print(f"   ✅ Created {col}_log with log(x + {constant}) transformation")
            
            elif fix_type == "Remove outliers":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                outliers_removed = (~outlier_mask).sum()
                
                # Note: This would modify the dataframe, so we just report
                self.ui.print(f"   📊 Would remove {outliers_removed} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
                self.ui.print("   💡 Apply this during preprocessing phase")
            
            elif fix_type == "Cap outliers at percentiles":
                p1 = df[col].quantile(0.01)
                p99 = df[col].quantile(0.99)
                df[f"{col}_capped"] = df[col].clip(lower=p1, upper=p99)
                self.ui.print(f"   ✅ Created {col}_capped with values capped at 1st-99th percentiles")
            
        except Exception as e:
            self.ui.print(f"   ❌ Error applying fix: {e}")
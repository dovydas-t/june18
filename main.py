import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration first
try:
    from config import *
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    VERSION = "3.0"
    APP_NAME = "Enhanced ML Pipeline"
    DEFAULT_MODELS = {'Random Forest': True, 'Linear Model': True}

def check_dependencies():
    """Check for required dependencies and provide installation instructions."""
    required = {
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0', 
        'sklearn': 'scikit-learn>=1.2.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'seaborn': 'seaborn>=0.11.0'
    }
    
    optional = {
        'rich': 'rich>=13.0.0 (for enhanced UI)',
        'xgboost': 'xgboost>=1.6.0 (for XGBoost model)',
        'lightgbm': 'lightgbm>=3.3.0 (for LightGBM model)'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, description in required.items():
        try:
            __import__(package)
        except ImportError:
            missing_required.append(description)
    
    for package, description in optional.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(description)
    
    if missing_required:
        print(f"❌ Missing required packages:")
        for pkg in missing_required:
            print(f"   {pkg}")
        print(f"\nInstall with: pip install {' '.join([p.split('>=')[0] for p in missing_required])}")
        return False
    
    if missing_optional:
        print(f"⚠️ Optional packages not found (reduced functionality):")
        for pkg in missing_optional:
            print(f"   {pkg}")
        print(f"\nInstall with: pip install {' '.join([p.split('>=')[0] for p in missing_optional])}")
    
    return True

# Try to import core components with fallbacks
try:
    from core.pipeline import EnhancedMLPipeline
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    CORE_AVAILABLE = False

def show_main_menu():
    """Display main menu options consistently."""
    print(f"\n🎯 Enhanced Machine Learning Pipeline v{VERSION}")
    print("="*50)
    print("Choose your mode:")
    print()
    print("1. 🖥️  Interactive Mode - Full pipeline with manual controls")
    print("2. 🚀 Quick Start Demo - Automated demo with sample data")
    print("3. 📚 Documentation & Features - Comprehensive guide")
    print("4. 🗄️  Database Browser - View experiment results")
    print("5. 🔧 System Check - Verify dependencies and configuration")
    print("6. ℹ️  About & Version Information - System info and credits")
    print()

def interactive_mode():
    """Enhanced interactive mode with better error handling."""
    if not CORE_AVAILABLE:
        print("❌ Enhanced pipeline not available.")
        print("Please install required dependencies first.")
        return
    
    try:
        if not check_dependencies():
            print("\n⚠️ Some required dependencies are missing.")
            print("The pipeline may not work correctly.")
            
            if not input("Continue anyway? (y/N): ").lower().startswith('y'):
                return
        
        print(f"\n🚀 Starting Enhanced ML Pipeline v{VERSION}...")
        
        pipeline = EnhancedMLPipeline()
        pipeline.main_menu()
        
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("• Check that all dependencies are properly installed")
        print("• Ensure you have write permissions in the current directory")
        print("• Try running with 'python3' instead of 'python'")
    finally:
        print("🗄️ Session ended.")

def quick_start_demo():
    """Quick start demo with sample data."""
    if not CORE_AVAILABLE:
        print("❌ Core pipeline not available. Please install dependencies first.")
        return
        
    print("🚀 QUICK START DEMO")
    print("="*30)
    print("This demo shows basic pipeline usage with synthetic data.")
    
    try:
        import pandas as pd
        import numpy as np
        
        print("\n📊 Creating sample dataset...")
        np.random.seed(42)
        
        n_samples = 1000
        data = {
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples), 
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature_4': np.random.uniform(0, 100, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('demo_data.csv', index=False)
        
        print(f"✅ Created demo_data.csv with {n_samples} samples")
        print("📋 Features: 2 numerical, 1 categorical, 1 continuous")
        print("🎯 Target: Binary classification")
        
        print(f"\n🚀 Starting demo pipeline...")
        
        pipeline = EnhancedMLPipeline(
            problem_type='classification',
            experiment_name='Quick_Start_Demo_v3'
        )
        
        success = pipeline.load_data('demo_data.csv', target_column='target')
        
        if success:
            print("✅ Demo data loaded successfully!")
            print("\nYou can now explore the v3.0 pipeline features:")
            print("• Modular architecture with specialized managers")
            print("• Enhanced data exploration engine") 
            print("• Improved preprocessing pipeline")
            print("• Better organized menu system")
            
            import os
            if os.path.exists('demo_data.csv'):
                os.remove('demo_data.csv')
                
        else:
            print("❌ Failed to load demo data")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def show_documentation():
    """Show comprehensive documentation."""
    print("📚 ENHANCED ML PIPELINE v3.0 DOCUMENTATION")
    print("="*55)
    
    print(f"\n🌟 NEW FEATURES IN v3.0:")
    print("• 🏗️ Completely refactored modular architecture")
    print("• 📦 Separated concerns into specialized managers")
    print("• 🔧 Enhanced preprocessing pipeline with specialized handlers")
    print("• 📊 Improved data exploration engine")
    print("• 🎛️ Better organized menu system")
    print("• 📈 Advanced metrics management system")
    print("• 🗄️ Enhanced database integration")
    print("• 🔍 Smart data analysis with AI recommendations")
    
    print(f"\n🏗️ MODULAR ARCHITECTURE:")
    print("• 📊 DataOperations: Centralized data loading and validation")
    print("• 🔍 ExplorationEngine: Comprehensive data analysis")
    print("• 📋 MenuManager: Organized navigation system")
    print("• 🔧 Preprocessing Modules: Specialized data preparation")
    print("• 🤖 Model Training: Enhanced model management")
    print("• 📈 Metrics Manager: Advanced performance evaluation")
    
    print(f"\n🤖 SUPPORTED MODELS:")
    print("• Instance-based: K-Nearest Neighbors")
    print("• Kernel-based: Support Vector Machines")
    print("• Tree-based: Decision Trees")
    print("• Linear: Linear/Logistic Regression")
    print("• Ensemble: Random Forest, Extra Trees, Bagging")
    print("• Boosting: AdaBoost, Gradient Boosting, XGBoost*, LightGBM*")
    print("  *Requires optional installation")

def show_system_check():
    """System check to verify dependencies and configuration."""
    print("🔧 SYSTEM CHECK v3.0")
    print("=" * 30)
    
    required_packages = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computing',
        'scikit-learn': 'Machine learning algorithms',
        'matplotlib': 'Plotting and visualization',
        'seaborn': 'Statistical visualization'
    }
    
    optional_packages = {
        'rich': 'Enhanced terminal UI',
        'xgboost': 'Advanced gradient boosting',
        'lightgbm': 'Fast gradient boosting'
    }
    
    print("📋 Required Packages:")
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description} [MISSING]")
    
    print("\n📦 Optional Packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️ {package} - {description} [OPTIONAL]")
    
    print("\n💾 System Information:")
    import sys
    print(f"  Python Version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    
    print("\n🏗️ Architecture Check:")
    print("  Checking modular components...")
    
    try:
        from core.data_operations import DataOperations
        print("  ✅ DataOperations module")
    except ImportError:
        print("  ❌ DataOperations module [MISSING]")
    
    try:
        from features.data_exploration.exploration_engine import ExplorationEngine
        print("  ✅ ExplorationEngine module")
    except ImportError:
        print("  ❌ ExplorationEngine module [MISSING]")
    
    try:
        from managers.menu_manager import MenuManager
        print("  ✅ MenuManager module")
    except ImportError:
        print("  ❌ MenuManager module [MISSING]")

def show_about_info():
    """Show comprehensive about and version information."""
    print("ℹ️ ABOUT & VERSION INFORMATION v3.0")
    print("=" * 50)
    print(f"🚀 {globals().get('APP_NAME', 'Enhanced ML Pipeline')} v{globals().get('VERSION', '3.0')}")
    print("📅 Release Date: 2024")
    print("👥 Authors: ML Pipeline Framework Team")
    print("")
    print("🔥 NEW IN VERSION 3.0:")
    print("• 🏗️ Complete architectural refactoring")
    print("• 📦 Modular design with separation of concerns")
    print("• 🔧 Specialized preprocessing handlers")
    print("• 📊 Enhanced data exploration capabilities")
    print("• 🎛️ Improved menu organization")
    print("• 📈 Advanced metrics system")
    print("• 🗄️ Better database integration")
    print("")
    print("🏗️ ARCHITECTURAL IMPROVEMENTS:")
    print("• Single Responsibility Principle applied")
    print("• Reduced file sizes for better maintainability")
    print("• Separated data operations from UI logic")
    print("• Modular preprocessing components")
    print("• Centralized menu management")
    print("• Enhanced error handling")
    print("")
    print("📄 LICENSE: MIT License")
    print("💻 GitHub: https://github.com/yourusername/enhanced-ml-pipeline")

def main():
    """Main execution function with enhanced error handling."""
    try:
        while True:
            show_main_menu()
            mode = input("Enter mode (1-6, or 0 to exit): ").strip()
            
            if mode == "0":
                print("👋 Goodbye!")
                break
            elif mode == "1":
                interactive_mode()
            elif mode == "2":
                quick_start_demo()
                input("\nPress Enter to return to main menu...")
            elif mode == "3":
                show_documentation()
                input("\nPress Enter to return to main menu...")
            elif mode == "4":
                print("🗄️ Database browser functionality moved to interactive mode")
                input("\nPress Enter to return to main menu...")
            elif mode == "5":
                show_system_check()
                input("\nPress Enter to return to main menu...")
            elif mode == "6":
                show_about_info()
                input("\nPress Enter to return to main menu...")
            else:
                print("❌ Invalid choice. Please enter 1-6 or 0 to exit.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Starting interactive mode...")
        interactive_mode()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For server environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class PinkCafeAI:
    """AI Model for Pink Cafe Sales Forecasting"""
    
    def __init__(self, filepath, product_name):
        self.filepath = filepath
        self.product_name = product_name
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and engineer features"""
        print(f"\n{'='*70}")
        print(f" LOADING: {self.product_name}")
        print(f"{'='*70}")
        
        self.df = pd.read_csv(self.filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df = self.df.rename(columns={'number_sold': 'sales'})
        
        # Time features
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 7, 14]:
            self.df[f'lag_{lag}'] = self.df['sales'].shift(lag)
        
        # Rolling features
        for window in [3, 7, 14]:
            self.df[f'roll_mean_{window}'] = self.df['sales'].rolling(window).mean()
            self.df[f'roll_std_{window}'] = self.df['sales'].rolling(window).std()
        
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df)} days")
        print(f"✓ Avg sales: {self.df['sales'].mean():.1f} units/day")
        print(f"✓ Features: {len(self.df.columns)-2}")
        
        return self.df
    
    def split_data(self, train_weeks=8, test_weeks=4):
        """Split into 8 weeks train, 4 weeks test"""
        print(f"\n{'='*70}")
        print(f"🔧 TRAIN/TEST SPLIT")
        print(f"{'='*70}")
        
        test_days = test_weeks * 7
        train_days = train_weeks * 7
        total = len(self.df)
        
        train_start = max(0, total - test_days - train_days)
        test_start = total - test_days
        
        train_df = self.df.iloc[train_start:test_start].copy()
        test_df = self.df.iloc[test_start:].copy()
        
        features = [c for c in self.df.columns if c not in ['date', 'sales']]
        
        X_train = train_df[features]
        y_train = train_df['sales']
        X_test = test_df[features]
        y_test = test_df['sales']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training: {len(X_train)} days ({train_weeks} weeks)")
        print(f"✓ Testing: {len(X_test)} days ({test_weeks} weeks)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df, train_df
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train 5 models"""
        print(f"\n{'='*70}")
        print(f"🤖 TRAINING MODELS")
        print(f"{'='*70}")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
        }
        
        results = []
        predictions = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            r2 = r2_score(y_test, y_pred)
            
            results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2})
            print(f"{name:22} MAPE: {mape:6.2f}%  MAE: {mae:5.2f}")
        
        results_df = pd.DataFrame(results).sort_values('MAPE')
        print(f"\n BEST: {results_df.iloc[0]['Model']} (MAPE: {results_df.iloc[0]['MAPE']:.2f}%)")
        
        return results_df, predictions
    
    def visualize(self, test_df, train_df, predictions, y_test, results_df):
        """Create 6-panel visualization"""
        print(f"\n{'='*70}")
        print(f"📈 CREATING VISUALIZATION")
        print(f"{'='*70}")
        
        fig = plt.figure(figsize=(20, 12))
        dates = test_df['date'].values
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        
        # 1. Time series
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(dates, y_test.values, 'o-', label='Actual', lw=3, ms=8, color='black', zorder=10)
        for idx, (name, pred) in enumerate(predictions.items()):
            ax1.plot(dates, pred, 'o--', label=name, lw=2, ms=5, alpha=0.7, color=colors[idx])
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Sales', fontweight='bold')
        ax1.set_title(f'{self.product_name} - 4-Week Forecast', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. MAPE bars
        ax2 = plt.subplot(2, 3, 2)
        bars = ax2.barh(range(len(results_df)), results_df['MAPE'], color=colors[:len(results_df)], alpha=0.7)
        bars[0].set_color('green')
        ax2.set_yticks(range(len(results_df)))
        ax2.set_yticklabels(results_df['Model'])
        ax2.set_xlabel('MAPE (%)', fontweight='bold')
        ax2.set_title('Performance (Lower=Better)', fontweight='bold', fontsize=13)
        ax2.grid(True, axis='x', alpha=0.3)
        for i, v in enumerate(results_df['MAPE']):
            ax2.text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # 3. Scatter
        ax3 = plt.subplot(2, 3, 3)
        best_name = results_df.iloc[0]['Model']
        best_pred = predictions[best_name]
        ax3.scatter(y_test, best_pred, alpha=0.7, s=120, edgecolors='black')
        min_val, max_val = min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect')
        ax3.set_xlabel('Actual', fontweight='bold')
        ax3.set_ylabel('Predicted', fontweight='bold')
        ax3.set_title(f'Best: {best_name}', fontweight='bold', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap
        ax4 = plt.subplot(2, 3, 4)
        heatmap = results_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2']].set_index('Model')
        sns.heatmap(heatmap.T, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax4, linewidths=1)
        ax4.set_title('Metrics Heatmap', fontweight='bold', fontsize=13)
        
        # 5. Residuals
        ax5 = plt.subplot(2, 3, 5)
        residuals = y_test.values - best_pred
        ax5.scatter(range(len(residuals)), residuals, alpha=0.7, s=100, c=residuals, cmap='RdYlGn_r')
        ax5.axhline(0, color='red', linestyle='--', lw=2)
        ax5.set_xlabel('Sample', fontweight='bold')
        ax5.set_ylabel('Residual', fontweight='bold')
        ax5.set_title(f'Residuals - {best_name}', fontweight='bold', fontsize=13)
        ax5.grid(True, alpha=0.3)
        
        # 6. Context
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(train_df['date'], train_df['sales'], '-', label='Training', lw=2, alpha=0.6)
        ax6.plot(dates, y_test.values, 'o-', label='Actual', lw=2.5, ms=8, color='black')
        ax6.plot(dates, best_pred, 'o--', label='Predicted', lw=2.5, ms=6, color='red')
        ax6.axvline(dates[0], color='green', linestyle='--', lw=2, label='Split')
        ax6.set_xlabel('Date', fontweight='bold')
        ax6.set_ylabel('Sales', fontweight='bold')
        ax6.set_title('Training Context', fontweight='bold', fontsize=13)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f'{self.product_name.replace(" ", "_")}_Analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

def main():
    print("\n" + "="*80)
    print(" "*20 + "PINK CAFÉ - AI MODEL DEVELOPMENT")
    print(" "*22 + "Student: Khadija Nisar")
    print("="*80)
    
    products = [
        ('americano_cleaned.csv', 'Americano Coffee'),
        ('cappuccino_cleaned.csv', 'Cappuccino Coffee'),
        ('croissant_cleaned.csv', 'Croissant')
    ]
    
    all_results = []
    
    for filepath, name in products:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {name}")
        print(f"{'='*80}")
        
        try:
            model = PinkCafeAI(filepath, name)
            model.load_data()
            X_train, X_test, y_train, y_test, test_df, train_df = model.split_data()
            results_df, predictions = model.train_models(X_train, X_test, y_train, y_test)
            model.visualize(test_df, train_df, predictions, y_test, results_df)
            
            # Save report
            report_name = f'{name.replace(" ", "_")}_Report.txt'
            best = results_df.iloc[0]
            
            report = f"""
================================================================================
PINK CAFÉ - {name.upper()} AI MODEL ANALYSIS
================================================================================

BEST MODEL: {best['Model']}
  • MAPE: {best['MAPE']:.2f}%
  • MAE:  {best['MAE']:.2f} units
  • RMSE: {best['RMSE']:.2f} units
  • R²:   {best['R2']:.4f}

ALL MODELS:
{results_df.to_string(index=False)}

INTERPRETATION:
On average, predictions are off by {best['MAE']:.2f} units per day.
This represents a {best['MAPE']:.2f}% error rate.
Model explains {best['R2']*100:.1f}% of variance in sales.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
            
            with open(report_name, 'w') as f:
                f.write(report)
            print(f"✓ Saved: {report_name}")
            
            results_df['Product'] = name
            all_results.append(results_df)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    if all_results:
        combined = pd.concat(all_results)
        combined.to_csv('All_Models_Results.csv', index=False)
        print(f"\n✓ Saved: All_Models_Results.csv")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  📊 3 PNG visualizations")
    print("  📄 3 TXT reports")
    print("  📈 1 CSV results file")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

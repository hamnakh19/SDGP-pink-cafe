import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    
    def __init__(self, filepath, product_name):
        self.filepath = filepath
        self.product_name = product_name
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):

        print(f"\n{'='*70}")
        print(f"📊 LOADING: {self.product_name}")
        print(f"{'='*70}")
        
        self.df = pd.read_csv(self.filepath)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df = self.df.rename(columns={'number_sold': 'sales'})
        
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        for lag in [1,2,3,7,14]:
            self.df[f'lag_{lag}'] = self.df['sales'].shift(lag)
        
        for window in [3,7,14]:
            self.df[f'roll_mean_{window}'] = self.df['sales'].rolling(window).mean()
            self.df[f'roll_std_{window}'] = self.df['sales'].rolling(window).std()
        
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df)} days")
        print(f"✓ Avg sales: {self.df['sales'].mean():.1f}")
        
        return self.df
    

    def split_data(self, train_weeks=8, test_weeks=4):

        print(f"\n{'='*70}")
        print(f"🔧 TRAIN TEST SPLIT")
        print(f"{'='*70}")
        
        test_days = test_weeks*7
        train_days = train_weeks*7
        total = len(self.df)
        
        train_start = max(0, total-test_days-train_days)
        test_start = total-test_days
        
        train_df = self.df.iloc[train_start:test_start].copy()
        test_df = self.df.iloc[test_start:].copy()
        
        features = [c for c in self.df.columns if c not in ['date','sales']]
        
        X_train = train_df[features]
        y_train = train_df['sales']
        X_test = test_df[features]
        y_test = test_df['sales']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training: {len(X_train)} days")
        print(f"✓ Testing: {len(X_test)} days")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df, train_df
    

    def train_models(self, X_train, X_test, y_train, y_test):

        print(f"\n{'='*70}")
        print(f"🤖 TRAINING MODELS")
        print(f"{'='*70}")
        
        models = {
            'Linear Regression':LinearRegression(),
            'Ridge Regression':Ridge(alpha=1.0),
            'Random Forest':RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42),
            'Gradient Boosting':GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=42),
            'SVR':SVR(kernel='rbf',C=100,epsilon=0.1)
        }
        
        results=[]
        predictions={}
        
        for name,model in models.items():
            
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            
            predictions[name]=y_pred
            
            rmse=np.sqrt(mean_squared_error(y_test,y_pred))
            mae=mean_absolute_error(y_test,y_pred)
            
            mask=y_test!=0
            mape=np.mean(np.abs((y_test[mask]-y_pred[mask])/y_test[mask]))*100
            
            r2=r2_score(y_test,y_pred)
            
            results.append({
                'Model':name,
                'RMSE':rmse,
                'MAE':mae,
                'MAPE':mape,
                'R2':r2
            })
            
            print(f"{name:22} MAPE:{mape:6.2f}%")
        
        results_df=pd.DataFrame(results).sort_values('MAPE')
        
        print(f"\n🏆 BEST MODEL: {results_df.iloc[0]['Model']}")
        
        return results_df,predictions
    

    def visualize(self, test_df, train_df, predictions, y_test, results_df):

        print(f"\n{'='*70}")
        print(f"📈 SAVING SEPARATE GRAPHS")
        print(f"{'='*70}")
        
        dates=test_df['date'].values
        colors=plt.cm.tab10(np.linspace(0,1,len(predictions)))
        
        # 1 Forecast
        plt.figure(figsize=(10,6))
        plt.plot(dates,y_test.values,'o-',label='Actual',lw=3,color='black')
        
        for idx,(name,pred) in enumerate(predictions.items()):
            plt.plot(dates,pred,'o--',label=name,color=colors[idx])
        
        plt.title(f'{self.product_name} - 4 Week Forecast')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Forecast.png',dpi=300)
        plt.close()
        
        # 2 Performance
        plt.figure(figsize=(10,6))
        plt.barh(results_df['Model'],results_df['MAPE'],color=colors)
        plt.title('Performance (Lower Better)')
        plt.xlabel('MAPE %')
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Performance.png',dpi=300)
        plt.close()
        
        # 3 Scatter
        best_name=results_df.iloc[0]['Model']
        best_pred=predictions[best_name]
        
        plt.figure(figsize=(8,6))
        plt.scatter(y_test,best_pred,s=120,edgecolors='black')
        
        min_val=min(y_test.min(),best_pred.min())
        max_val=max(y_test.max(),best_pred.max())
        
        plt.plot([min_val,max_val],[min_val,max_val],'r--')
        plt.title(f'Best Model: {best_name}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Scatter.png',dpi=300)
        plt.close()
        
        # 4 Heatmap
        heatmap=results_df[['Model','RMSE','MAE','MAPE','R2']].set_index('Model')
        
        plt.figure(figsize=(8,6))
        sns.heatmap(heatmap.T,annot=True,fmt='.2f',cmap='RdYlGn_r')
        plt.title('Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Heatmap.png',dpi=300)
        plt.close()
        
        # 5 Residuals
        residuals=y_test.values-best_pred
        
        plt.figure(figsize=(10,6))
        plt.scatter(range(len(residuals)),residuals,c=residuals,cmap='RdYlGn_r')
        plt.axhline(0,color='red',linestyle='--')
        plt.title('Residuals')
        plt.xlabel('Sample')
        plt.ylabel('Residual')
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Residuals.png',dpi=300)
        plt.close()
        
        # 6 Training Context
        plt.figure(figsize=(10,6))
        plt.plot(train_df['date'],train_df['sales'],label='Training')
        plt.plot(dates,y_test.values,'o-',label='Actual',color='black')
        plt.plot(dates,best_pred,'o--',label='Predicted',color='red')
        plt.axvline(dates[0],color='green',linestyle='--')
        plt.title('Training Context')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.product_name}_Context.png',dpi=300)
        plt.close()


def main():

    print("="*80)
    print("PINK CAFE AI MODEL")
    print("="*80)
    
    products=[
        ('americano_cleaned.csv','Americano Coffee'),
        ('cappuccino_cleaned.csv','Cappuccino Coffee'),
        ('croissant_cleaned.csv','Croissant')
    ]
    
    for filepath,name in products:
        
        print(f"\nPROCESSING {name}")
        
        model=PinkCafeAI(filepath,name)
        
        model.load_data()
        
        X_train,X_test,y_train,y_test,test_df,train_df=model.split_data()
        
        results_df,predictions=model.train_models(X_train,X_test,y_train,y_test)
        
        model.visualize(test_df,train_df,predictions,y_test,results_df)
        
        report_name=f'{name}_Report.txt'
        
        best=results_df.iloc[0]
        
        report=f"""
BEST MODEL: {best['Model']}

MAPE:{best['MAPE']:.2f}
MAE:{best['MAE']:.2f}
RMSE:{best['RMSE']:.2f}
R2:{best['R2']:.4f}

{results_df}
"""
        
        with open(report_name,'w') as f:
            f.write(report)


if __name__=="__main__":
    main()
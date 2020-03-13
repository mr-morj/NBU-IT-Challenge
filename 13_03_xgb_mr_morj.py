import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
from sklearn import preprocessing
from xgboost import plot_importance

def corr_matrix(anotation):
    
    corr = data_full.corr()
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, annot=anotation)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
    plt.title('Correlation Heatmap')

def scaling(data):
    
    plt.plot(data)
    plt.title('Without Scaling')
    plt.show()
    
    mm_scaler = preprocessing.MinMaxScaler()
    data_minmax = data.copy()
    scaled_features = ['avg_sal_uah', 'fx_reserves', 'gdp_pers_uah', 'gdp_pers_usd', 'fdi', 'state_debt_uah', 'state_debt_usd']
    data_minmax[scaled_features] = mm_scaler.fit_transform(data[scaled_features])
    return data_minmax 


df = pd.read_excel('final_data.xlsx', parse_date=True, index_col='date')

df.drop(df.tail(1).index,inplace=True)
#df['gdp_pers_usd'] = df['gdp_pers_usd'].interpolate(method='linear')
df = df.interpolate(method='polynomial', order=3)
df = scaling(df)
    

df.to_excel('interpolate_data.xlsx')

data_usd = pd.read_csv('exchange_rate.csv')
data_usd['date'] = pd.to_datetime(data_usd['date'], format='%d.%m.%Y', errors='ignore')
data_usd = data_usd.set_index('date')
data_usd = data_usd.resample('M').mean()
data_usd = data_usd[:'2019-12-31']

data_full = df.copy()
data_full.index = np.arange(len(data_full))
data_usd_copy = data_usd.copy()
data_usd_copy.index = np.arange(len(data_usd_copy))
data_full['exchange'] = data_usd_copy['ex_rate']

corr_matrix(False)

potential_features = ['ppi', 'inflation', 'avg_sal_uah', 'ipi', 'fx_reserves', 'agro_ppi',
       'gdp_pers_uah', 'gdp_pers_usd', 'unemployment', 'bop', 'rsi', 'fdi',
       'gov_budg', 'int_trade', 'state_debt_uah', 'state_debt_usd']

X = df[potential_features]
X_train = X[:96]
y_train = data_usd[:96]
X_test = X[96:]
y_test = data_usd[96:]




model = xgb.XGBRegressor()
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False)

_ = plot_importance(model, height=0.8)

y_predict = model.predict(X_test)
plt.show()
y_copy = y_test.copy()
y_copy.loc[:,'ex_rate'] = y_predict

plt.plot(data_usd, 'b', label='real data')
plt.plot(y_copy, 'y', label='predict data')
plt.legend()
plt.title('Values')
plt.show()





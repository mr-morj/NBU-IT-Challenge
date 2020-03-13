import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

#warnings.filterwarnings("ignore")

def interpolation_plot(column_name):
    
    data_lin=data_raw[column_name].interpolate(method='linear')
    data_pol=data_raw[column_name].interpolate(method='polynomial', order=2)

    fig, axs = plt.subplots(2, figsize=(9,9))
    fig.suptitle('Interpolation comparison')
    
    axs[0].plot(data_raw.index, data_lin, 'y-', label='linear')
    axs[0].scatter(data_raw.index, data_raw[column_name], color='r', s=20)
    
    axs[1].plot(data_raw.index, data_pol, 'y-', label='polynomial')
    axs[1].scatter(data_raw.index, data_raw[column_name], color='r', s=20)
    
    axs[0].legend(loc=1)
    axs[1].legend(loc=1)

def corr_matrix(anotation):
    
    corr = data.corr()
    plt.subplots(figsize=(10,8))
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                cmap=sns.diverging_palette(0, 200, n=200), 
                annot=anotation)
    plt.title('Correlation Heatmap of Numeric Features')

def feat_plot(model):
    
    feat_plot = plot_importance(model, height=0.9)
    feat_plot.figure.set_size_inches(10, 8)
    
def mean_absolute_percentage_error(y_true, y_pred): 
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs( (y_true - y_pred)/y_true)) * 100

def predict_plot():
    
    fig, axs = plt.subplots(1, figsize=(9,7))
    fig.suptitle('Predictions/real values')
    axs.plot(data.index, data.exrate, 'b-', label='real')
    axs.plot(X_test.index, preds, 'r-', label='prediction')
    axs.legend(loc=2)

data = pd.read_excel('final_data.xlsx', parse_dates=True, index_col='date')

data=data[:-1]
data_raw=data.copy()

data=data.interpolate(method='polynomial', order=2)
data_usd = pd.read_csv('exchange_rate.csv')

data_usd['date'] = pd.to_datetime(data_usd['date'], format='%d.%m.%Y', errors='ignore')
data_usd = data_usd.set_index('date')
data_usd = data_usd['exrate']

data_usd = data_usd.resample('M').mean()
data_usd = data_usd[:'2019-12-31']

data = data.assign(exrate = data_usd.values)
data = data['2015-02-01':]
#tss = TimeSeriesSplit(n_splits = 4)
tss = KFold(n_splits = 5)

X = data.drop(labels=['exrate'], axis=1)
y = data['exrate']

for train_index, test_index in tss.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

model = XGBRegressor()
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)   


preds = model.predict(X_test)

interpolation_plot('bop')
corr_matrix(False)
feat_plot(model)
predict_plot()    
print('MAPE: {}%'.format(mean_absolute_percentage_error(y_test,
                                                        preds).round(3)))

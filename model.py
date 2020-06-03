import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import datetime
from xgboost import XGBRegressor
# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMAResults
#from IPython.display import display, Markdown
# import folium 
# from folium import plugins

plt.rcParams['figure.figsize'] = 15, 12
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit,RandomizedSearchCV

from joblib import dump,load


## Reading the dataset

total_data= pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv',parse_dates=True)
total_data_ind=total_data[total_data['location']=='India']

total_data_ind.tail()

total_data_ind.columns

df=total_data_ind[['date','total_cases']]

df.columns=['Date','Confirmed']

# df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',parse_dates=True)
# df.shape

# df.drop('Sno',axis=1,inplace=True)

# df.drop(['ConfirmedIndianNational','ConfirmedForeignNational'],axis=1,inplace=True)

# df.dropna(how='any',axis=0,inplace=True)

# df['Cured']=df['Cured'].apply(lambda x: int(x))
# df['Deaths']=df['Deaths'].apply(lambda x: int(x))


current_date=df['Date'].iat[-1]
# total_so_far = df[df['Date']==current_date]['Confirmed'].sum()
# print(f"Total cases in India as of {current_date} are {total_so_far}")

# ## Statewise distribution of cases

# df['active_cases']=df['Confirmed']-df['Cured']-df['Deaths']
# df[df['Date']==current_date][['State/UnionTerritory','Cured','Deaths','Confirmed']].sort_values('Confirmed',ascending=False).style.background_gradient(cmap='Reds').hide_index()


# ## Active cases distribution

# df[df['Date']==current_date][['State/UnionTerritory','active_cases']].sort_values('active_cases',ascending=False).rename(columns= {'active_cases':'Active Cases'}).style.background_gradient(cmap='Reds').hide_index()

# ## Confirmed vs Cured figures

# f, ax = plt.subplots(figsize=(12, 10))
# data = df[df['Date']==current_date][['State/UnionTerritory','Cured','Deaths','Confirmed','active_cases']]
# data.sort_values('Confirmed',ascending=False,inplace=True)
# sns.set_color_codes("pastel")
# sns.barplot(x="Confirmed", y="State/UnionTerritory", data=data,label="Total", color="r")

# sns.set_color_codes("muted")
# sns.barplot(x="Cured", y="State/UnionTerritory", data=data, label="Cured", color="g")

# max_cases=data['Confirmed'].iat[0]

# ax.legend(ncol=2, loc="lower right", frameon=True)
# ax.set(xlim=(0, max_cases), ylabel="",xlabel="Cases")
# sns.despine(left=True, bottom=True)

datewise_df = df.copy()



	


## Analyzing the trend in India

def analyze_trend():
	
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=datewise_df['Date'], y = datewise_df['Confirmed'], mode='lines+markers',name='Total Cases'))
	fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',xaxis_title='Date',yaxis_title='Confirmed Cases',plot_bgcolor='rgb(230, 230, 230)')
	fig.show()


	# Log Scale Trend 

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=datewise_df['Date'], y = np.log10(datewise_df['Confirmed']), mode='lines+markers',name='Total Cases in Log Scale'))
	fig.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases)',xaxis_title='Date',yaxis_title='Confirmed Cases in Log10 Scale',plot_bgcolor='rgb(230, 230, 230)')
	fig.show()


	fig = px.bar(datewise_df, x="Date", y="Confirmed", barmode='group', height=400)
	fig.update_layout(title_text='Coronavirus Cases in India on daily basis',plot_bgcolor='rgb(230, 230, 230)')

	fig.show()

	fig = px.bar(total_data_ind, x=total_data_ind["date"], y=total_data_ind["new_tests"], barmode='group', height=400)
	fig.update_layout(title_text='New tests in India',xaxis_title='Date',yaxis_title='New Tests',plot_bgcolor='rgb(230, 230, 230)')
	fig.show()





days = np.array([i for i in range(len(datewise_df['Date']))]).reshape(-1,1)
confirmed_cases=np.array(datewise_df['Confirmed']).reshape(-1,1)
# cured_cases=np.array(datewise_df['Cured']).reshape(-1,1)
# death_cases=np.array(datewise_df['Deaths']).reshape(-1,1)

days_in_future=20
future_forecast = np.array([i for i in range(len(datewise_df['Date'])+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-days_in_future]


start=datewise_df['Date'].iat[0]
start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
actual_df.to_csv('actual_df.csv',index=False)

X_train_confirmed,X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days,confirmed_cases,test_size=0.25,shuffle=False,stratify=None)


def train_model_LR():

	param_grid_lr={'poly__degree':[4,5,7,8,9,10]}
	pipeline = Pipeline(steps=[('poly', PolynomialFeatures()), ('ridge', Ridge())])
	tscv = TimeSeriesSplit(n_splits=2)

	grid_search = GridSearchCV(pipeline, param_grid_lr, cv=tscv,scoring='neg_mean_squared_error',return_train_score=True)
	grid_search.fit(X_train_confirmed, y_train_confirmed)
	test_linear_pred=grid_search.predict(X_test_confirmed)
	
	print("MAE: ",mean_absolute_error(y_test_confirmed,test_linear_pred))
	print("RMSE: ",sqrt(mean_squared_error(y_test_confirmed,test_linear_pred)))
	
	plt.plot(y_test_confirmed)
	plt.plot(test_linear_pred)
	plt.legend(['Test Data',f'Polynomial regression with d={grid_search.best_params_["poly__degree"]}'])
	# plt.show()

	lr_model=grid_search.best_estimator_
	dump(lr_model,'lr_model.joblib')

## Predicting for future
def show_LR():
	lr_model=load('lr_model.joblib')
	future_linear_pred=lr_model.predict(future_forecast)
	pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_linear_pred).reshape(-1,)})
	

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	fig.show()
	return f"The number of cases might reach upto {int(pred_df['Cases'].iat[-1])} on {pred_df['Date'].iat[-1]} as per LinearRegression predictor"



# X_train_confirmed_2=X_train_confirmed.copy()
# s=StandardScaler()
# s.fit(X_train_confirmed_2)
# X_train_confirmed_2=s.transform(X_train_confirmed_2)



def train_model_svr():
	#X_train_confirmed,X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days,confirmed_cases,test_size=0.25,shuffle=False,stratify=None)
	
	param_grid_svr = {
	    "kernel": ["rbf","poly"],
	    "C": [1e5,1e7,1e8],
	    "gamma": [1e-8,1e-4,1e-5,1e-9],
	    "epsilon":[1e-5,1e-8,1e-3],
	    "shrinking":[False,True],
	    "degree":[2,5,7]
	    }
	tscv = TimeSeriesSplit(n_splits=2)
	grid_search_svr = GridSearchCV(SVR(),param_grid=param_grid_svr,cv=tscv,scoring='neg_mean_squared_error')
	grid_search_svr.fit(X_train_confirmed,y_train_confirmed)
	test_svr_pred=grid_search_svr.predict(X_test_confirmed)
	print("MAE: ",mean_absolute_error(y_test_confirmed,test_svr_pred))
	print("RMSE: ",sqrt(mean_squared_error(y_test_confirmed,test_svr_pred)))
	# grid_search.best_params_
	# plt.plot(y_test_confirmed)
	# plt.plot(test_svr_pred)
	# plt.legend(['Test Data','SVR regression'])
	# plt.show()
	svr_model=grid_search_svr.best_estimator_
	dump(svr_model,'svr_model.joblib')


# def train_xgb():
# 	m=XGBRegressor(n_estimators=1000,objective= "reg:squarederror")
# 	m.fit(X_train_confirmed,y_train_confirmed)
# 	test_xgb_pred=m.predict(X_test_confirmed)
# 	print("MAE: ",mean_absolute_error(y_test_confirmed,test_xgb_pred))
# 	print("RMSE: ",sqrt(mean_squared_error(y_test_confirmed,test_xgb_pred)))
# 	future_xgb_pred=m.predict(future_forecast)
# 	pred_df_xgb=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_xgb_pred).reshape(-1,)})
# 	pred_df_xgb.to_csv('pred_df_xgb.csv')






def show_model_svr():

	svr_model=load('svr_model.joblib')
	future_svr_pred=svr_model.predict(future_forecast)
	pred_df_svr=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_svr_pred).reshape(-1,)})
	actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=pred_df_svr['Date'], y = pred_df_svr['Cases'] , mode='lines+markers',name='Prediction using SVR',line={'color':'red'}))
	fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(title_text='Prediction of Coronavirus Cases in India (SVR)',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	fig.show()


	return f"The number of cases might reach upto {int(pred_df_svr['Cases'].iat[-1])} on {pred_df_svr['Date'].iat[-1]} as per support vector regression predictor"



# autocorrelation_plot(new_df)
# plt.show()
# model = ARIMA(new_df, order=(10,1,0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())
def train_arima():
	new_df=pd.Series(actual_df['Cases'].values,index=actual_df['Date'])
	X=new_df.values
	# size = int(len(X) * 0.66)
	# train, test = X[0:size], X[size:len(X)]
	# history = [x for x in train]
	# predictions = list()
	model = ARIMA(X, order=(10,1,0))
	model_fit = model.fit()
	
	model_fit.save('arima_model.pkl')
	# for t in range(len(test)):
	# 	model = ARIMA(history, order=(10,1,0))
	# 	model_fit = model.fit()
	# 	output = model_fit.forecast()
	# 	yhat = output[0]
	# 	predictions.append(yhat)
	# 	obs = test[t]
	# 	history.append(obs)
	# 	print('predicted=%f, expected=%f' % (yhat, obs))
	# error = sqrt(mean_squared_error(test, predictions))
	# print('Test MSE: %.3f' % error)
	# # plot
	# plt.plot(test)
	# plt.plot(predictions, color='red')
	# plt.show()

# train_model_LR()
# train_model_svr()
# train_arima()
train_xgb()
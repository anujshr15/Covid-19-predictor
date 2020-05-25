from flask import Flask, jsonify , request,render_template
import numpy as np
import pandas as pd 
from joblib import load,dump
import flask
import datetime
import io
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__,template_folder="./")

@app.route('/')
def hello_world():
	return "Covid-19 Predictor"

@app.route('/index')
def index():
	return render_template("index.html")


@app.route('/train_LR',methods=['GET'])
def show_LR():

	actual_df=pd.read_csv('actual_df.csv',parse_dates=True)
	days_in_future=20
	future_forecast = np.array([i for i in range(len(actual_df['Date'])+days_in_future)]).reshape(-1,1)
	start=actual_df['Date'].iat[0]
	start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
	future_forecast_dates = []
	for i in range(len(future_forecast)):
		future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
	lr_model=load("lr_model.joblib")
	future_linear_pred=lr_model.predict(future_forecast)
	pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_linear_pred).reshape(-1,)})
	#actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	# img = io.BytesIO()
	fig = go.Figure()
	t1=fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	t2=fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	# data=[t1,t2]
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('index.html',graphJSON=graphJSON)
	

	# img.seek(0)
	# plot_url = base64.b64encode(img.getvalue()).decode()
	# return '<img src="data:image/png;base64,{}">'.format(plot_url)
	#return jsonify({'prediction':f"The number of cases might reach upto {int(pred_df["Cases"].iat[-1])} on {pred_df["Date"].iat[-1]} as per LinearRegression predictor"})


@app.route('/train_SVR',methods=['GET'])
def show_SVR():

	actual_df=pd.read_csv('actual_df.csv',parse_dates=True)
	days_in_future=20
	future_forecast = np.array([i for i in range(len(actual_df['Date'])+days_in_future)]).reshape(-1,1)
	start=actual_df['Date'].iat[0]
	start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
	future_forecast_dates = []
	for i in range(len(future_forecast)):
		future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))
	lr_model=load("svr_model.joblib")
	future_svr_pred=lr_model.predict(future_forecast)
	pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_svr_pred).reshape(-1,)})
	#actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	# img = io.BytesIO()
	fig = go.Figure()
	t1=fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	t2=fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	# data=[t1,t2]
	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('index.html',graphJSON=graphJSON)



if __name__=='__main__':
	app.run(host='0.0.0.0',port=8080)

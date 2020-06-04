from flask import Flask, jsonify , request,render_template
import numpy as np
import pandas as pd 
from joblib import load,dump
import flask
import datetime
import io
import json
import time,random
import base64
import requests
from urllib.request import Request, urlopen
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

actual_df=pd.read_csv('actual_df.csv',parse_dates=True)
days_in_future=20
future_forecast = np.array([i for i in range(len(actual_df['Date'])+days_in_future)]).reshape(-1,1)
start=actual_df['Date'].iat[0]
start_date = datetime.datetime.strptime(start,'%Y-%m-%d')
future_forecast_dates = []
for i in range(len(future_forecast)):
	future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d"))




app = Flask(__name__,template_folder="./")

@app.route('/')
def hello_world():

	time.sleep(random.randint(0, 2))  # relax and don't let google be angry
	r = requests.get("https://news.google.com/search?q=coronavirus+india")
	print(r.status_code)  # Print the status code
	content = r.text
	news_summaries = []
	links=[]
	bsoup = soup(content, "html.parser")
	st_divs = bsoup.find_all('h3')[:7]
	# st_links= bsoup.find_all('a',href=True)[:7]
	for st_div in st_divs:
		news_summaries.append(st_div.text)
	for news in st_divs:
		for link in news.find_all('a'):
			links.append("https://news.google.com"+link.get('href')[1:])

	# data=pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv',parse_dates=True)
	# confirmed=data['Total Confirmed'].iat[-1]
	# recovered=data['Total Recovered'].iat[-1]
	# deaths=data['Total Deceased'].iat[-1]
	df=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv',parse_dates=True)
	confirmed=df.loc[0,'Confirmed']
	recovered=df.loc[0,'Recovered']
	deaths = df.loc[0,'Deaths']

	return render_template("index.html",**locals())

@app.route('/index')
def index():
	return render_template("index.html")


@app.route('/train_LR',methods=['GET'])
def show_LR():


	lr_model=load("lr_model.joblib")
	future_linear_pred=lr_model.predict(future_forecast)
	pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_linear_pred).reshape(-1,)})
	#actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	# img = io.BytesIO()
	fig = go.Figure()
	t1=fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	t2=fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',autosize=True,title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	# data=[t1,t2]
	graphJSON = json.dumps([fig], cls=plotly.utils.PlotlyJSONEncoder)
	prediction="The number of cases might reach upto  "+str(int(pred_df["Cases"].iat[-1]))+" on "+ str(pred_df["Date"].iat[-1])+" as per Ridge Regression predictor"
	return render_template('index.html',graphJSON=graphJSON,prediction=prediction)
	

	# img.seek(0)
	# plot_url = base64.b64encode(img.getvalue()).decode()
	# return '<img src="data:image/png;base64,{}">'.format(plot_url)
	#return jsonify({'prediction':f"The number of cases might reach upto {int(pred_df["Cases"].iat[-1])} on {pred_df["Date"].iat[-1]} as per LinearRegression predictor"})


@app.route('/train_SVR',methods=['GET'])
def show_SVR():


	lr_model=load("svr_model.joblib")
	future_svr_pred=lr_model.predict(future_forecast)
	pred_df=pd.DataFrame({'Date':pd.Series(future_forecast_dates),'Cases':np.array(future_svr_pred).reshape(-1,)})
	#actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	# img = io.BytesIO()
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(autosize=True,    paper_bgcolor='rgba(0,0,0,0)',title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases',plot_bgcolor='rgb(230, 230, 230)')
	
	# data=[t1,t2]
	graphJSON = json.dumps([fig], cls=plotly.utils.PlotlyJSONEncoder)
	prediction="The number of cases might reach upto  "+str(int(pred_df["Cases"].iat[-1]))+" on "+ str(pred_df["Date"].iat[-1])+" as per Support Vector Regression predictor"

	return render_template('index.html',graphJSON=graphJSON,prediction=prediction)




@app.route('/train_arima',methods=['GET'])
def show_arima():


	pred_df=pd.read_csv('arima_pred.csv',parse_dates=True)
	#actual_df= pd.DataFrame({'Date':np.array(future_forecast_dates[:-days_in_future]).reshape(-1,),'Cases':confirmed_cases.reshape(-1,)})
	# img = io.BytesIO()
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=pred_df['Date'], y = pred_df['Cases'] , mode='lines+markers',name='Prediction',line={'color':'red'}))
	fig.add_trace(go.Scatter(x=actual_df['Date'], y =actual_df['Cases'], mode='lines+markers',name='Actual so far',line={'color':'blue'}))
	fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',autosize=True,title_text='Prediction of Coronavirus Cases in India',xaxis_title='Date',yaxis_title='Corona Virus Cases')
	# data=[t1,t2]
	graphJSON = json.dumps([fig], cls=plotly.utils.PlotlyJSONEncoder)
	prediction="The number of cases might reach upto  "+str(int(pred_df["Cases"].iat[-1]))+" on "+ str(pred_df["Date"].iat[-1])+" as per ARIMA predictor"

	return render_template('index.html',graphJSON=graphJSON,prediction=prediction)





@app.route('/state_wise',methods=['GET'])
def state_wise_trend():
	df=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv',parse_dates=True)
	fig = go.Figure(data=[go.Table(header=dict(values=df.columns[:5], fill_color='paleturquoise'),
                 cells=dict(values=[df[df.columns[i]] for i in range(0,5)],
				 fill_color='lavender'))

                     ])
	fig.update_layout(autosize=True,height=1200,margin=dict(t=30, b=0, l=0, r=0))

	graphJSON = json.dumps([fig],cls=plotly.utils.PlotlyJSONEncoder)
	



	return render_template('index.html',graphJSON=graphJSON)












@app.route('/show_graphs',methods=['GET'])
def analyze_trend():
	datewise_df=pd.read_csv('actual_df.csv',parse_dates=True)
	
	fig1 = go.Figure()
	fig1.add_trace(go.Scatter(x=datewise_df['Date'], y = datewise_df['Cases'], mode='lines+markers',name='Total Cases'))
	fig1.update_layout(autosize=True,title_text='Trend of Coronavirus Cases in India (Cumulative cases)',xaxis_title='Date',yaxis_title='Confirmed Cases',plot_bgcolor='rgb(230, 230, 230)')
	# fig1.show()


	# Log Scale Trend 

	fig2 = go.Figure()
	fig2.add_trace(go.Scatter(x=datewise_df['Date'], y = np.log10(datewise_df['Cases']), mode='lines+markers',name='Total Cases in Log Scale'))
	fig2.update_layout(title_text='Trend of Coronavirus Cases in India (Cumulative cases) in Log Scale',xaxis_title='Date',yaxis_title='Confirmed Cases in Log10 Scale',plot_bgcolor='rgb(230, 230, 230)')
	# # fig2.show()

	fig3 = go.Figure()
	fig3 = px.bar(datewise_df, x="Date", y="Cases", barmode='group', height=400)
	fig3.update_layout(title_text='Coronavirus Cases in India on daily basis',plot_bgcolor='rgb(230, 230, 230)')

	# fig3.show()

	# fig4 = px.bar(total_data_ind, x=total_data_ind["date"], y=total_data_ind["new_tests"], barmode='group', height=400)
	# fig4.update_layout(title_text='New tests in India',xaxis_title='Date',yaxis_title='New Tests',plot_bgcolor='rgb(230, 230, 230)')
	# fig4.show()
	
	graphJSON = json.dumps([fig1,fig2,fig3], cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('index.html',graphJSON=graphJSON)




if __name__=='__main__':
	app.run(host='0.0.0.0',port=8080)

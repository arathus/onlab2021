import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append("/home/imetomi/Projects/Trading-Algorithms")
sys.path.append("C:/Users/Imetomi/Documents/MEGA/Projects/Trading-Algorithms")
from backtest.backtest import Backtest
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import random

import chart_studio.plotly as py	
import plotly.graph_objs as go
from plotly import subplots
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
pd.options.display.float_format = '{:.5f}'.format

class Portofolio:
	def __init__(self, instruments, strategies, distribution, leverages, directs, resolution, from_date, to_date, indicators=[], balance=10000, ipynb=False, ddw=10):
		self.Colors = ['#1f77b4', '#3366cc', '#4c78a8 ', '#316395', '#0099c6', '#2e91e5', '#0d2a63', '#3366cc']
		self.Strategies = strategies
		self.Instruments = instruments
		self.Distribution = distribution
		self.Direct = directs
		self.Leverages = leverages
		self.FromDate = from_date
		self.Balance = balance
		self.ToDate = to_date
		self.Resolutions = resolution
		self.Indicators = indicators
		self.Results = []
		self.Analysis = pd.DataFrame()
		self.Logs = None
		self.DDW = ddw
		self.GrossLoss = None
		self.GrossProfit = None
		self.ipynb = ipynb
		self.BM = True
		self.RfR = 0.02
		snp_benchmark = None											# loading S&P as benchmark
		dji_benchmark = None											# loading DJI as benchmark
		dax_benchmark = None											# loading DAX as benchmark
		self.DJI_Benchmark = None
		self.SNP_Benchmark = None
		self.DAX_Benchmark = None

		if self.BM:
			try:
				snp_benchmark = pd.read_csv('data/datasets/spx500usd/spx500usd_hour.csv')
			except:
				snp_benchmark = pd.read_csv('../data/datasets/spx500usd/spx500usd_hour.csv')

			try:
				dji_benchmark = pd.read_csv('data/datasets/djiusd/djiusd_hour.csv')
			except:
				dji_benchmark = pd.read_csv('../data/datasets/djiusd/djiusd_hour.csv')

			try:
				dax_benchmark = pd.read_csv('data/datasets/de30eur/de30eur_hour.csv')
			except:
				dax_benchmark = pd.read_csv('../data/datasets/de30eur/de30eur_hour.csv')

			self.DJI_Benchmark = self.section(dji_benchmark, str(self.FromDate).split(' ')[0], str(self.ToDate).split(' ')[0])
			self.SNP_Benchmark = self.section(snp_benchmark, str(self.FromDate).split(' ')[0], str(self.ToDate).split(' ')[0])
			self.DAX_Benchmark = self.section(dax_benchmark, str(self.FromDate).split(' ')[0], str(self.ToDate).split(' ')[0])


	def section(self, dt, from_date, to_date):
		start, end = None, None
		try:
			start = dt.index[dt['Date'] == from_date].tolist()[0]
		except:
			print('[ERROR] Incorrect starting date.')
		try:
			end = dt.index[dt['Date'] == to_date].tolist()
			end = end[len(end) - 1]
		except:
			print('[ERROR] Incorrect ending date.')
		return dt[start:end].reset_index()



	def max_dd(self, data_slice):
		max2here = data_slice.expanding().max()
		dd2here = data_slice - max2here
		return dd2here.min()


	def length(self, df):
		if len(df) > 0:
			return len(df)
		else:
			return 1



	def analyze(self):
		if len(self.Results) > 0:
			if self.DDW != 0:
				self.Results['Drawdown'] = self.Results['Balance'].rolling(self.DDW).apply(self.max_dd)
			else:
				dd_length = len(self.Data) / len(self.Results)
				elf.Results['Drawdown'] = self.Results['Balance'].rolling(dd_length).apply(self.max_dd)

			columns = ['Nr. of Trades', 'Profit / Loss', 'Profit Factor', 'Win Ratio', 'DDW', 'DDW (%)', 'Sharpe Ratio', 'Balance', 'Max. Balance', 'Min. Balance', 'Balance Std.', 'Gross Profit', 'Gross Loss', 'Winning Trades', 'Losing Trades', 'Average P/L', 'P/L Std.', 'Average Profit', 'Average Loss', 'Profit Std.', 'Loss Std.', 'SL/TP Activated', 'RoR (10%)', 'RoR (20%)', 'RoR (30%)', 'Outlier Sensitivity']
			
			self.GrossLoss = self.Results[self.Results['Profit'] <= 0]['Profit'].sum()
			self.GrossProfit = self.Results[self.Results['Profit'] >= 0]['Profit'].sum()
			if self.GrossLoss == 0:
				self.GrossLoss = 1
			
			buy = self.Results[self.Results['Type'] == 'BUY']
			buy_values = [len(buy), buy['Profit'].sum(), buy[buy['Profit'] > 0]['Profit'].sum() / abs(buy[buy['Profit'] < 0]['Profit'].sum()), 
					len(buy[buy['Profit'] > 0]) / self.length(buy), None, None, None, None, None, None, None, 
					buy[buy['Profit'] > 0]['Profit'].sum(), buy[buy['Profit'] < 0]['Profit'].sum(),
					len(buy[buy['Profit'] > 0]), len(buy[buy['Profit'] < 0]), 
					buy['Profit'].sum() / self.length(buy), buy['Profit'].std(),
					buy.loc[buy['Profit'] > 0]['Profit'].mean(), buy.loc[buy['Profit'] < 0]['Profit'].mean(), 
					buy.loc[buy['Profit'] > 0]['Profit'].std(), buy.loc[buy['Profit'] < 0]['Profit'].std(), 
					buy['AutoClose'].sum(), None, None, None, None]

			sell = self.Results[self.Results['Type'] == 'SELL']
			sell_values = [len(sell), sell['Profit'].sum(), sell[sell['Profit'] > 0]['Profit'].sum() / abs(sell[sell['Profit'] < 0]['Profit'].sum()), 
					len(sell[sell['Profit'] > 0]) / self.length(sell), None, None, None, None, None, None, None,
					sell[sell['Profit'] > 0]['Profit'].sum(), abs(sell[sell['Profit'] < 0]['Profit'].sum()), 
					len(sell[sell['Profit'] > 0]), len(sell[sell['Profit'] < 0]), sell['Profit'].sum() / self.length(sell), sell['Profit'].std(),
					sell.loc[sell['Profit'] > 0]['Profit'].mean(), sell.loc[sell['Profit'] < 0]['Profit'].mean(), 
					sell.loc[sell['Profit'] > 0]['Profit'].std(), sell.loc[sell['Profit'] < 0]['Profit'].std(), 
					sell['AutoClose'].sum(), None, None, None, None]
			

			sharpe_ratio = (self.Results['Balance'][len(self.Results)-1] / self.Balance - 1 - self.RfR) / (self.Results['Balance'] / self.Balance).std()

			wr = len(self.Results[self.Results['Profit'] > 0]) / len(self.Results)
			lr = 1 - wr 
			avg_loss = abs(self.Results.loc[self.Results['Profit'] < 0]['Profit'].mean())
			avg_win = abs(self.Results.loc[self.Results['Profit'] > 0]['Profit'].mean())
			c = self.Balance
			avg_loss_percent = avg_loss / c
			avg_win_percent = avg_win / c  
			z = (wr * avg_win_percent) - (lr * avg_loss_percent)
			a = (wr * avg_win_percent ** 2 + lr * avg_loss_percent ** 2) ** (1/2)
			p = 0.5 * (1 + z/a)
			ror = abs((1-p)/p)
			
			RoR10 = min(ror ** (0.1 / a), 1)
			RoR20 = min(ror ** (0.2 / a), 1)
			RoR30 = min(ror ** (0.3 / a), 1)
			profit = self.Results.sort_values(by='Profit', ascending=False)
			OLS = (profit[profit['Profit'] > 0][:int(np.ceil(len(profit) * 0.1))]['Profit'].sum()) / (profit[profit['Profit'] > 0][int(np.ceil(len(profit) * 0.1)):]['Profit'].sum())

			
			all_values = [len(self.Results), sum(self.Results['Profit']), self.GrossProfit / abs(self.GrossLoss),  wr, 
					self.Results['Drawdown'].min(), abs(self.Results['Drawdown'].min()) / self.Results['Balance'].max(), sharpe_ratio, self.Results['Balance'].iloc[-1], self.Results['Balance'].max(), self.Results['Balance'].min(), self.Results['Balance'].std(), self.GrossProfit, self.GrossLoss, 
					len(self.Results[self.Results['Profit'] > 0]), len(self.Results[self.Results['Profit'] < 0]), self.Results['Profit'].mean(), self.Results['Profit'].std(),
					self.Results.loc[self.Results['Profit'] > 0]['Profit'].mean(),  self.Results.loc[self.Results['Profit'] < 0]['Profit'].mean(), 
					self.Results.loc[self.Results['Profit'] > 0]['Profit'].std(), self.Results.loc[self.Results['Profit'] < 0]['Profit'].std(), 
					self.Results['AutoClose'].sum(), RoR10, RoR20, RoR30, OLS]
			

			self.Analysis['Ratio'] = columns
			self.Analysis['All'] = all_values
			self.Analysis['Long'] = buy_values
			self.Analysis['Short'] = sell_values



	def run(self):
		if len(self.Strategies) != len(self.Instruments) or len(self.Strategies) != len(self.Distribution) or len(self.Strategies) != len(self.Leverages):
			print("[ERROR] Different parameter lengths! Stopping simulation.")
			return
		for i in range(len(self.Strategies)):
			data = pd.read_csv('data/datasets/' + self.Instruments[i] + '/' + self.Instruments[i] + '_' + self.Resolutions[i] + '.csv')
			backtest = Backtest(self.Strategies[i], data, self.FromDate, self.ToDate, leverage=self.Leverages[i], balance=self.Balance * self.Distribution[i], 
								direct=self.Direct[i], ddw=10)
			for ind in self.Indicators:
				backtest.add_indicator(ind[0], ind[1])
			
			print(self.Instruments[i].upper())
			backtest.run()
			self.Results.append(backtest.TradeLog)

		self.Logs = copy.deepcopy(self.Results)
		self.Results = pd.concat(self.Results)
		self.Results = self.Results.sort_values(by='Close Time').reset_index().drop('index', axis=1)
		self.Results['Balance'][0] = self.Balance
		for i in range(1, len(self.Results)):
			self.Results['Balance'][i] = self.Balance + sum(self.Results['Profit'][:i])
		self.analyze()



	def plot_results(self, name='portofolio_results.html'):
		if len(self.Results) == 0:
			print("No trades were made")
			return
		
		buysell_color = []
		entry_shape = []
		profit_color = []
		for _, trade in self.Results.iterrows():
			if trade['Type'] == 'BUY':
				buysell_color.append('#83ccdb')
				entry_shape.append('triangle-up')
			else:
				buysell_color.append('#ff0050')
				entry_shape.append('triangle-down')
			if trade['Profit'] > 0:
				profit_color.append('#cdeaf0')
			else:
				profit_color.append('#ffb1cc')


		fig = subplots.make_subplots(rows=3, cols=3, column_widths=[0.55, 0.27, 0.18],
									specs=[[{}, {}, {"rowspan": 2, "type": "table"}], 
											[{}, {}, None], 
											[{}, {"type": "table", "colspan": 2}, None]],
									shared_xaxes=True,
									subplot_titles=("Balance", "Benchmarks", "Performance Analysis", "Drawdown", "Monte Carlo Simulation", "Profit and Loss", "List of Trades"), 
									vertical_spacing=0.06, horizontal_spacing=0.02)

		buysell_marker = dict(color=buysell_color, size=self.Results['Profit'].abs() / self.Results['Profit'].abs().max() * 40)

		bubble_plot = go.Scatter(x=self.Results['Close Time'], y=self.Results['Profit'], name='P/L',
									marker=buysell_marker, mode='markers',
									hovertemplate = '<i>P/L</i>: %{y:.5f}' + '<b>%{text}</b>',
									text='<br>Market: ' + self.Results['Instrument'] + 
										 '<br>ID: ' + self.Results.index.astype(str) + 
										 '<br>OP: ' + self.Results['Open Price'].astype(str) +
										 '<br>CP: ' + self.Results['Close Price'].astype(str) +
										 '<br>OT: ' + self.Results['Open Time'] +
										 '<br>CT: ' + self.Results['Close Time'] +
										 '<br>Units: ' + self.Results['Units'].astype(str))

		drawdown_plot = go.Scatter(x=pd.concat([pd.Series([self.Results['Open Time'][0]]), self.Results['Close Time']]), 
										y=pd.concat([pd.Series([0]), self.Results['Drawdown']]), 
										name='DDW' + ' ' + str(self.DDW), connectgaps=True, fill='tozeroy', line_color="#ff0050", mode='lines')

		profit_plot = go.Scatter(x=self.Results['Close Time'], y=self.Results['Profit'], name='Profit', 
									connectgaps=True, marker=dict(color='#1d3557'))

		balance_plot = go.Scatter(x=pd.concat([pd.Series([self.Results['Open Time'][0]]), self.Results['Close Time']]), 
										y=pd.concat([pd.Series([self.Balance]), self.Results['Balance']]), 
										name='Balance', connectgaps=True, fill='tozeroy', line_color="#5876F7", mode='lines')

		result_list = go.Table(header=dict(values=['Ratio', 'All', 'Long', 'Short']), 
								   cells=dict(values=[self.Analysis['Ratio'], self.Analysis['All'], self.Analysis['Long'], self.Analysis['Short']],
								   			  format=[None] + [".2f"],
											  height = 40,
											  fill = dict(color=['#C7D4E2', ' #EAF0F8'])))

		not_needed = ['TP', 'SL', 'AutoClose', 'Drawdown', 'Spread']
		trade_list = go.Table(header=dict(values=self.Results.drop(not_needed, axis=1).columns), 
							cells=dict(values=[self.Results.drop(not_needed, axis=1)[column] for column in self.Results.drop(not_needed, axis=1)], 
							format=[None] * 4 + [".2f"] * 2 + [".5f"] * 3 + [".2f"] * 2,
							font=dict(color=['rgb(40, 40, 40)'] * 10, size=11),
							fill_color=[profit_color * 10]))
											 
		benchmark_plot = go.Scatter(x=pd.concat([pd.Series(self.Results['Open Time'][0]), self.Results['Close Time']]), 
										y=pd.concat([pd.Series([self.Balance]), self.Results['Balance']]) / self.Balance - 1, name='BM', 
										connectgaps=True, line_color="#5876F7")

		
		# calculating monte carlo simulation
		balance_reference_plot = go.Scatter(x=[x for x in range(len(self.Results))], y=self.Results['Balance'], name='MC Ref', line_color="#5876F7", mode='lines')
		last_balance = self.Results['Balance'][len(self.Results)-1]
		avg = (self.Results['Profit'].sum() / len(self.Results)) / self.Balance
		std_dev = self.Results['Profit'].std() / self.Balance
		num_reps = int(len(self.Results) / 2)
		num_simulations = 10
		avg_at = 2
		monte_carlos = []
		for x in range(num_simulations):
			price_series = [last_balance]
			price = last_balance * (1 + np.random.normal(0, std_dev))
			price_series.append(price)
			for y in range(num_reps):
				price = price_series[len(price_series)-1] * (1 + np.random.normal(0, std_dev))
				price_series.append(price)
			monte_carlos.append(np.array(price_series))
			if len(monte_carlos) >= avg_at:
				monte_carlos = np.array(monte_carlos)
				summed = sum(monte_carlos)
				monte_carlo = summed / len(monte_carlos)
				monte_carlo_plot = go.Scatter(x=[x+len(self.Results)-1 for x in range(len(monte_carlo))], y=monte_carlo, name='MC', mode='lines')
				fig.append_trace(monte_carlo_plot, 2, 2)
				monte_carlos = []

		if self.BM:
				self.SNP_Benchmark = self.SNP_Benchmark.iloc[::int(len(self.SNP_Benchmark) / len(self.Results)), :]
				snp_plot = go.Scatter(x=self.SNP_Benchmark['Date'] + ' ' + self.SNP_Benchmark['Time'], y=self.SNP_Benchmark['AC'] / self.SNP_Benchmark['AC'][0] - 1, name='S&P', 
										connectgaps=True, marker=dict(color='#b7c0fa'))

				self.DJI_Benchmark = self.DJI_Benchmark.iloc[::int(len(self.DJI_Benchmark) / len(self.Results)), :]
				dji_plot = go.Scatter(x=self.DJI_Benchmark['Date'] + ' ' + self.DJI_Benchmark['Time'], y=self.DJI_Benchmark['AC'] / self.DJI_Benchmark['AC'][0] - 1, name='DJI', 
										connectgaps=True, marker=dict(color='#F35540'))

				self.DAX_Benchmark = self.DAX_Benchmark.iloc[::int(len(self.DAX_Benchmark) / len(self.Results)), :]
				dax_plot = go.Scatter(x=self.DAX_Benchmark['Date'] + ' ' + self.DAX_Benchmark['Time'], y=self.DAX_Benchmark['AC'] / self.DAX_Benchmark['AC'][0] - 1, name='DAX', 
										connectgaps=True, marker=dict(color='#FECB52'))

				fig.append_trace(snp_plot, 1, 2)
				fig.append_trace(dji_plot, 1, 2)
				fig.append_trace(dax_plot, 1, 2)
		
		
		
		for i in range(len(self.Instruments)):
			instrument_plot = go.Scatter(x=pd.concat([pd.Series([self.Logs[i]['Open Time'][0]]), self.Logs[i]['Close Time']]), 
							y=pd.concat([pd.Series([self.Balance * self.Distribution[i]]), self.Logs[i]['Balance']]), 
							name=self.Instruments[i].upper(), connectgaps=True, mode='lines', line_color=self.Colors[i])
			fig.append_trace(instrument_plot, 1, 1)

		fig.append_trace(trade_list, 3, 2)
		fig.append_trace(balance_reference_plot, 2, 2)
		fig.append_trace(benchmark_plot, 1, 2)
		fig.append_trace(balance_plot, 1, 1)
		fig.append_trace(result_list, 1, 3)
		fig.append_trace(drawdown_plot, 2, 1)
		fig.append_trace(bubble_plot, 3, 1)
		fig.append_trace(profit_plot, 3, 1)


		fig.update_layout(xaxis_rangeslider_visible=False, title=go.layout.Title(text = 'Portofolio Results'))
		if self.ipynb:
			iplot(fig)
		else:
			plot(fig, filename=name)


		
		


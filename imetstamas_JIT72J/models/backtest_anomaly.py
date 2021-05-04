# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append('/home/imetomi/Minyma/Minyma-DEV')
sys.path.append('C:/Users/Imetomi/Documents/MEGA/Minyma/Minyma-DEV')

# %%
signals = pd.read_csv('data/entries.csv')


# %%
signals


# %%
cleaned_data = pd.read_csv('data/lob_cleaned.csv')


# %%
cleaned_data


# %%
from_index = int(len(cleaned_data) * 0.7 + 51096 - 50979)


# %%
data = pd.DataFrame(columns=['Date', 'Time', 'AO', 'AH', 'AL', 'AC', 'BO', 'BH', 'BL', 'BC', 'Entry'])
data.Date = [date[0] for date in cleaned_data.date.str.split(' ')][from_index:]
data.Time = [date[1] for date in cleaned_data.date.str.split(' ')][from_index:]
data.AC = cleaned_data.ask[from_index:].reset_index().ask
data.AO = cleaned_data.ask[from_index:].reset_index().ask
data.AL = cleaned_data.ask[from_index:].reset_index().ask
data.AH = cleaned_data.ask[from_index:].reset_index().ask
data.BC = cleaned_data.bid[from_index:].reset_index().bid
data.BH = cleaned_data.bid[from_index:].reset_index().bid
data.BL = cleaned_data.bid[from_index:].reset_index().bid
data.BO = cleaned_data.bid[from_index:].reset_index().bid
data.Entry = signals.entry


# %%
len(signals)


# %%
data


# %%
from backtest.backtest import Backtest


# %%
from backtest.strategy import Strategy
from backtest.backtest import Backtest
from backtest.analysis import Analysis


prev_idx = 0

# %%
class DLModel(Strategy):
	def __init__(self, *args, **kwargs):
		self.prev_profit = 0
		self.trailing = 1.45 / 100
		super(DLModel, self).__init__(*args, **kwargs)

	def execute(self, backtest, row, idx):
		global prev_idx
		if backtest.Data.Entry[idx] > 0.92:
			if idx - prev_idx > 600:
				prev_idx = idx
				backtest.close(row, 0)
				backtest.buy(row, self.Instrument, self.TradeAmmount, self.StopLoss * backtest.Balance, self.TakeProfit * backtest.Balance)


# %%
leverage = 1
trade_amount = 0.5
symbol = 'btcusdt'


# %%
strat = DLModel(0.005 * leverage * trade_amount, 0.0, symbol.upper(), trade_amount)


# %%
import matplotlib.pyplot as plt

plt.plot(cleaned_data.ask[from_index:].reset_index().ask)


# %%
backtest = Backtest(strat, datetime(2021, 3, 1), datetime(2021, 3, 1), data, direct=False, leverage=leverage, balance=1000, commission=0.04, max_units=200000000)
backtest.run()


# %%
backtest.TradeLog


# %%
analysis = Analysis(backtest, sortino=0.6, log=True)
analysis.analyze()
analysis.plot('results.html')


# %%




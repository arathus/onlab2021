import numpy as np
import pandas as pd

class PseudoLimitOrderBook:
    def __init__(self, data, time):
        self.time = time
        self.data = [data]
        self.pd = ""

    def __add__(self, data):
        self.data +=  [data]
        return self

    def cls_lob(self):
        cols = np.concatenate((["time"], ["a"+str(i) for i in range(0,40)], ["b"+str(i) for i in range(0,40)]))
        self.pd = pd.DataFrame(self.data, columns=cols)
        return self.pd

class PseudoTradeData:
    def __init__(self, data, timestamp):
        self.timestamp = timestamp
        self.num_of_trades = 1
        self.wprices = np.array([data["price"]*data["volume"]])
        self.sent_wprices = np.array([data["price"]*data["volume"] if data["mmaker"] else -data["price"]*data["volume"]])
        self.volume = data["volume"]

    def __add__(self, data):
        self.num_of_trades += 1
        self.wprices = np.append(self.wprices, data["price"]*data["volume"])
        self.sent_wprices = np.append(self.sent_wprices, data["price"]*data["volume"] if data["mmaker"]\
            else -data["price"]*data["volume"])
        self.volume += data["volume"]
        return self

    def cls_trdata(self):
        return {"timestamp": self.timestamp, 
            "num_of_trades": self.num_of_trades,
            "quant_weighted_avg_price": float(np.sum(self.wprices)/self.volume), 
            "sentiment_weighted_avg_price": float(np.sum(self.sent_wprices)/self.volume)}

class PseudoCandleData:
    def __init__(self, value, volume, timestamp):
        self.timestamp = timestamp
        self.open, self.high, self.low, self.close = [value]*4
        self.volume  = volume

    def __add__(self, data):
        self.high = self.high if data["value"] < self.high else data["value"]
        self.low = self.low if data["value"] > self.low else data["value"]
        self.close = data["value"]
        self.volume += data["volume"]
        return self

    def cls_candle(self):
        self.type = "Green" if self.close > self.open else "Red"
        return {"timestamp": self.timestamp, "open":self.open, "high": self.high, "low": self.low, "close":self.close, \
        "volume":self.volume, "type":self.type}

class PseudoPriceData:
    def __init__(self, data, timestamp):
        self.timestamp = timestamp
        self.last_price, self.last_quantity = data["lp"], data["lq"]
        self.best_ask, self.best_ask_q = data["ba"], data["baq"]
        self.best_bid, self.best_bid_q = data["bb"], data["bbq"]
        self.volume = data["vol"]

    def __add__(self, data):
        self.last_price, self.last_quantity = data["lp"], data["lq"]
        self.best_ask = data["ba"] if self.best_ask < data["ba"] else self.best_ask
        self.best_ask_q = data["baq"] if self.best_ask < data["ba"] else self.best_ask_q
        self.best_bid = data["bb"] if self.best_bid > data["bb"] else self.best_bid
        self.best_bid_q = data["bbq"] if self.best_bid > data["bbq"] else self.best_bid_q
        self.volume += data["vol"]
        return self
        
    def close_ppirce(self):
        return {'timestamp': self.timestamp,
         'last_price': self.last_price,
         'last_quant': self.last_quantity, 
         'best_ask_price': self.best_ask,
         'best_ask_quant': self.best_ask_q,
         'best_bid_price': self.best_bid,
         'best_bid_quant': self.best_bid_q,
         'all_volume': self.volume}
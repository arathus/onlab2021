from binance.client import Client
from binance.websockets import BinanceSocketManager
from keys import api, secret
from binance.enums import *
import os
import json 
import time 

client = Client(api, secret)
bm = BinanceSocketManager(client, user_timeout=60)

main_path = '/mnt/hdd/'
kline_path = 'kline'
trades_path = 'trades'
lob_path = 'lob'
btc_path = 'btc'
eth_path = 'eth'

btc_depth_dict = {}
eth_depth_dict = {}

btc_ticker_dict = {}
eth_ticker_dict = {}

btc_trades_dict = {}
eth_trades_dict = {}


def process_depth(msg):
    global btc_depth_dict, eth_depth_dict
    if msg['e'] == 'depthUpdate':
        if msg['s'] == 'BTCUSDT':
            instance = {msg['E'] : {'b' : msg['b'], 'a': msg['a']}}
            btc_depth_dict.update(instance)
            if len(btc_depth_dict) > 60:
                with open(os.path.join(lob_path, btc_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(btc_depth_dict, file)
                btc_depth_dict = {}

        elif msg['s'] == 'ETHUSDT':
            instance = {msg['E'] : {'b' : msg['b'], 'a': msg['a']}}
            eth_depth_dict.update(instance)
            if len(eth_depth_dict) > 60:
                with open(os.path.join(lob_path, eth_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(eth_depth_dict, file)
                eth_depth_dict = {}
    

def process_kline(msg):
    global btc_ticker_dict, eth_ticker_dict
    if msg['e'] == '24hrTicker':
        if msg['s'] == 'BTCUSDT':
            instance = {msg['E']: msg}
            btc_ticker_dict.update(instance)

            if len(btc_ticker_dict) > 60:
                with open(os.path.join(kline_path, btc_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(btc_ticker_dict, file)
                btc_ticker_dict = {}

        elif msg['s'] == 'ETHUSDT':
            instance = {msg['E']: msg}
            eth_ticker_dict.update(instance)
            if len(eth_ticker_dict) > 60:
                with open(os.path.join(kline_path, eth_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(eth_ticker_dict, file)
                eth_ticker_dict = {}


prev_time_btc = time.time()
prev_time_eth = time.time()

def minute_passed(check_time):
    return time.time() - check_time >= 60

def process_trades(msg):
    global eth_trades_dict, btc_trades_dict, prev_time_btc, prev_time_eth
    if msg['e'] == 'trade':
        if msg['s'] == 'BTCUSDT':
            btc_trades_dict.update({msg['T'] : msg})
            if minute_passed(prev_time_btc):
                prev_time_btc = time.time()
                with open(os.path.join(trades_path, btc_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(btc_trades_dict, file)
                btc_trades_dict = {}

        if msg['s'] == 'ETHUSDT':
            eth_trades_dict.update({msg['T'] : msg})
            if minute_passed(prev_time_eth):
                prev_time_eth = time.time()
                with open(os.path.join(trades_path, eth_path, str(msg['E']) + '.json'), 'w') as file:
                    json.dump(eth_trades_dict, file)
                eth_trades_dict = {}


depth_key_btc = bm.start_depth_socket('BTCUSDT', process_depth)
depth_key_eth = bm.start_depth_socket('ETHUSDT', process_depth)

kline_key_btc = bm.start_symbol_ticker_socket('BTCUSDT', process_kline)
kline_key_eth = bm.start_symbol_ticker_socket('ETHUSDT', process_kline)

trades_key_btc = bm.start_trade_socket('BTCUSDT', process_trades)
trades_key_eth = bm.start_trade_socket('ETHUSDT', process_trades)

bm.start()

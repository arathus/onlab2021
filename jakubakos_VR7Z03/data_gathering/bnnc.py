from binance.client import Client
from binance.websockets import BinanceSocketManager
from _creds import api_key, api_secret
from pprint import pprint
import sys, json, time, os
from datetime import date, datetime

def write_json(data, symbol, filename):

    if not os.path.exists(f'./{symbol}/{filename}.json'):
        data = {"order_books": [data]}

    with open(f'./{symbol}/{filename}.json','w') as f: 
        json.dump(data, f, indent=4)
        f.close()

def save_order_book(symbol, client, filename):
    ordr = client.get_order_book(symbol=symbol, limit=5000)

    if not os.path.exists(f'./{symbol}'):
        os.makedirs(f"./{symbol}")

    ordr["req_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    desired_order_list = ["req_time", "lastUpdateId", "bids", "asks"]

    reordered_dict = {k: ordr[k] for k in desired_order_list}

    if os.path.exists(f'./{symbol}/{filename}.json'):
        with open(f'./{symbol}/{filename}.json') as json_file: 
            data = json.load(json_file) 
            temp = data['order_books']
            temp.append(reordered_dict)
            json_file.close()
    else:
        data = reordered_dict

    write_json(data, symbol, filename)

def getFolderSize(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += getFolderSize(itempath)

    if total_size > 1024**3:
        return f'Size of {folder}: {format(total_size/1024**3, ".4f")} GB'
    elif total_size > 1024**2:
        return f'Size of {folder}: {format(total_size/1024**2, ".4f")} MB'
    elif total_size > 1024:
        return f'Size of {folder}: {format(total_size/1024, ".4f")} KB'
    else:
        return f'Size of {folder}: {format(total_size, ".4f")} byte'

def draw_information():
    draw.rectangle((0,0,width,height), outline=0, fill=0)
    cmd = "top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}'"
    cpu_usage = subprocess.check_output(cmd, shell = True )
    cmd = "free -m | awk 'NR==2{printf \"%.2f%%\", $3/$2 }'"
    mem_usage = subprocess.check_output(cmd, shell = True )
    draw.text((x, top+20), "MEM: " + str(mem_usage)[2:6] + "   CPU: " + str(cpu_usage)[2:6], font=font, fill=255)
    draw.text((x, top+30),  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=font, fill=255)
    draw.text((x, top+40),  getFolderSize("BTCUSDT"), font=font, fill=255)
    draw.text((x, top+50),  getFolderSize("ETHUSDT"), font=font, fill=255)
    draw.text((x, top+60),  getFolderSize("LINKUSDT"), font=font, fill=255)
    disp.image(image)
    disp.display()

client = Client(api_key, api_secret)
starttime = time.time()
sleep_in_secs = 60*5
minute_counter = 0

while True:
    if minute_counter % 360 is 0:
        filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_order_book("BTCUSDT", client, filename)
    save_order_book("ETHUSDT", client, filename)
    save_order_book("LINKUSDT", client, filename)

    minute_counter +=1
    time.sleep(sleep_in_secs - ((time.time() - starttime) % sleep_in_secs))

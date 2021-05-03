import json
import numpy as np
import pandas as pd
from os import walk
from pprint import pprint
from tqdm.auto import tqdm
from datetime import datetime, timedelta
from dateutil.parser import parse
from pathlib import Path


path = "../data/trades/eth/"
_, _, filenames = next(walk(path))

for idx, item in enumerate(filenames):
    filenames[idx] = path + item

with open(filenames[-1]) as json_file:
    data = json.load(json_file)
    elements = [data[k] for k in list(data.keys())]

pprint(elements)
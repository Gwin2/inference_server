# this module will be imported in the into your flowgraph

# from gnuradio import gr
import time
import numpy as np
import os
import psutil
import sys
import requests
import json
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
server_host = os.getenv('server_ip')
server_port = os.getenv('server_port')
PARAMS = {'split_size': 1_000_000}
token = 1

    ##############################
    # HYPERPARAMETERS
    ##############################

f_base = 2.48e9
f_step = -10e6
f_roof = 2.4e9

    # p2p_border = np.array([0.05, 0.065, 0.08, 0.95, 0.115, 0.125, 0.135, 0.185, 0.25, 0.3])

    ##############################
    # Variables
    ##############################

# median_vals = []
f = f_base  # local frequency
EOCF = 0  # End of changing frequency flag
signal_arr = []

    ##############################
    # support functions
    ##############################

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def send_data(sig):
    global token
    data_to_send = {
        "freq": 2400,
        "data_real": np.asarray(np.array(sig, dtype=np.complex64).real, dtype=np.float32),
        "data_imag": np.asarray(np.array(sig, dtype=np.complex64).imag, dtype=np.float32),
        "token": token
    }
    mod_data_to_send = json.dumps(data_to_send, cls=NumpyArrayEncoder)
    response = requests.post("http://{0}:{1}/receive_data".format('192.168.1.90', '8080'), json=mod_data_to_send)
    if response.status_code == 200:
        print('#' * 10)
        print('TOKEN ' + str(token))
        token += 1
        print(response.text)
        print('#' * 10)
    else:
        print('#' * 10)
        print("Ошибка при отправке данных: ", response.status_code)
        print('#' * 10)

def work(lvl):

    global f_base
    global f_step
    global f_roof
    global f
    global EOCF
    global signal_arr

    y = np.array(lvl).ravel()
    signal_arr = np.concatenate((signal_arr, y), axis=None)

    if f <= f_roof:
        f = f_base
        signal_arr = []
        # send_data(np.max(np.array(median_vals)))
        # median_vals = []
        return f, EOCF
    else:
        if len(signal_arr) >= PARAMS['split_size']:
            send_data(signal_arr[:PARAMS['split_size']])
            # m = calc_median(sig)
            # median_vals.append(m)
            # print(m)
            signal_arr = []
            f += f_step
            # time.sleep(10)
        return f, EOCF
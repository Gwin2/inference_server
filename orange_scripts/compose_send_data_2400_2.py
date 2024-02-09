from dotenv import load_dotenv
import numpy as np
import requests
import os
import sys
import json


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
server_ip = os.getenv('SERVER_IP')
server_port = os.getenv('SERVER_PORT')
num_token = os.getenv('NUM_TOKEN')
PARAMS = {'split_size': 2_000_000, 'split_chank': 400_000}
token = 0

##############################
# HYPERPARAMETERS
##############################
f_base = 2.5e9
f_step = -20e6
f_roof = 2.4e9
ind = 1
##############################
# Variables
##############################
f = f_base
EOCF = 0
signal_arr = []


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def send_data(sig):
    try:
        global token
        print('#' * 10)
        print('\nОтправка пакета ' + str(token+1))
        data_to_send = {
            "freq": 2400,
            "token": int(token+1),
            "data_real": np.asarray(np.array(sig, dtype=np.complex64).real, dtype=np.float32),
            "data_imag": np.asarray(np.array(sig, dtype=np.complex64).imag, dtype=np.float32)
        }
        mod_data_to_send = json.dumps(data_to_send, cls=NumpyArrayEncoder)
        response = requests.post("http://{0}:{1}/receive_data".format(server_ip, server_port), json=mod_data_to_send)
        if response.status_code == 200:
            token += 1
            print(response.text)
            print('#' * 10)
        else:
            print("Ошибка при отправке данных: ", response.status_code)
            print('#' * 10)
        if int(num_token) == token:
            sys.exit()
    except Exception as exc:
        print(str(exc))
        sys.exit()


def work(lvl):
    global f_base
    global f_step
    global f_roof
    global f
    global EOCF
    global signal_arr
    global length

    y = np.array(lvl).ravel()
    print(len(y))
    signal_arr = np.concatenate((signal_arr, y), axis=None)

    if f <= f_roof:
        f = f_base
        signal_arr = []
        return f, EOCF
    else:
        if len(signal_arr) >= PARAMS['split_size']:
            send_data(signal_arr[:PARAMS['split_size']])
            ind = 1
            signal_arr = []
        elif len(signal_arr) >= PARAMS['split_chank']*ind:
            signal_arr = signal_arr[:PARAMS['split_chank']*ind]
            ind+=1
            f += f_step
        return f, EOCF
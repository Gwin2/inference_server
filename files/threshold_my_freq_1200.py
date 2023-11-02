# this module will be imported in the into your flowgraph
from gnuradio import gr
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

##############################
# RKNN
##############################

PARAMS = {
'split_size': 100_000,
}

##############################
# HYPERPARAMETERS
##############################

f_base = 1.36e9
f_step = -20e6
f_roof  = 1.16e9

p2p_border = np.array([0.05, 0.065, 0.08, 0.95, 0.115, 0.125, 0.135, 0.185, 0.25, 0.3])

##############################
# Variables
##############################

median_vals = []
f = f_base       # local frequency
EOCF = 0         # End of changing frequency flag
signal_arr = []
##############################
# support functions
##############################

def calc_median(sig):
    m = np.abs(sig)
    m = np.median(m)
    return m


def send_data(sig_dist):
    len_threshold = int(os.getenv('len_threshold'))
    localhost = os.getenv('lochost')
    localport = os.getenv('locport')
    length = int(np.sum(np.where(p2p_border <= sig_dist, 1, 0)))

    if length >= len_threshold:
        trigger = True
    else:
        trigger = False

    data_to_send = {
                    "freq": 1200,
                    "amplitude": 55,
                    "triggered": trigger,
                    "light_len": length
                    }

    response = requests.post("http://{0}:{1}/process_data".format(localhost, localport), json=data_to_send)

    if response.status_code == 200:
        print("Данные успешно отправлены и приняты!")
    else:
        print("Ошибка при отправке данных: ", response.status_code)

##############################
# main function
##############################

def work(lvl):

    global f_base
    global f_step
    global f_roof

    global f

    global median_vals
    global EOCF
    global signal_arr

    y = np.array(lvl).ravel()
    signal_arr = np.concatenate((signal_arr, y), axis=None)

    if f <= f_roof:
        f = f_base
        signal_arr = []
        send_data(np.max(np.array(median_vals)))
        median_vals = []
        return f, EOCF
    else:
        if len(signal_arr) >= PARAMS['split_size']:
            sig = np.array([signal_arr.real, signal_arr.imag])
            m = calc_median(sig)
            median_vals.append(m)
            #print(m)
            signal_arr = []
            f += f_step
        return f, EOCF

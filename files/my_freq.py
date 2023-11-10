
# this module will be imported in the into your flowgraph
from gnuradio import gr
import time
from scipy.fftpack import fft, fftfreq
import numpy as np
from scipy.signal import find_peaks, spectrogram
from scipy.signal.windows import hann
#import matplotlib.pyplot as plt
import os
import psutil
import sys

import platform
from rknnlite.api import RKNNLite

import wiringpi as wpi
from wiringpi import GPIO

import requests
import json
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
##############################
# wiringPi
##############################

wpi.wiringPiSetup()

##############################
# RKNN
##############################

DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

RK3588_MODEL = {
'path': os.getenv('path_to_NN'),
'model': None,
'split_size': 50_000,
'N_predictions': 40,
'N_samples_confidence_threshold': 0.65,
}

RK3588_MODEL['model'] = RKNNLite()

try:
	ret1 = RK3588_MODEL['model'].load_rknn(RK3588_MODEL['path'])
	assert(ret1 == 0)
except Exception as e:
	print(e)

ret2 = RK3588_MODEL['model'].init_runtime(core_mask=RKNNLite.NPU_CORE_0)
assert(ret2 == 0)

##############################
# HYPERPARAMETERS
##############################

f_base = 2.48e9
f_step = -10e6
f_roof  = 2.4e9

signal_num = 10
#signal_dir = '/home/orangepi/CDD/Complex_DroneDetection/gnuradio/signal/'
#if not os.path.exists(signal_dir):
#    os.mkdir(signal_dir)


reading_signal_delay = 0
iterations = 3  # read signal iterations
height_threshold = 100

weak_avg_amount = 70
weak_samples_confidence = 0.50
#classes = {0: 'noise', 1: 'DJI', 2: 'other', 3: 'for weak'}
classes = {0: 'noise', 1: 'DJI_video', 2: 'DJI_control', 3: 'WIFI'}

#pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 7]
pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 7]

on_state = 0
off_state = 1
p2p_border = np.array([0.05, 0.065, 0.08, 0.95, 0.115, 0.125, 0.135, 0.185, 0.25, 0.3])
amp_border = np.array([0, 0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
#amp_decays = np.array([0, -0.15, -0.14, -0.13, -0.12, -0.11, -0.10,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01, -0.009, -0.008, -0.007])
amp_decays = np.array([0, -0.07, -0.0675, -0.065, -0.0625, -0.06, -0.05,-0.045,-0.04,-0.035,-0.03,-0.025,-0.02,-0.015,-0.01,-0.008, -0.004, -0.002, -0.001])

assert(len(amp_border) == len(amp_decays))
amp_slice_size = 50000
pred_amps = []
##############################
# Variables
##############################

running_amp = 0

weak_detected = 0
counter = 0
ctrs = {0: 0, 1: 0, 2: 0, 3: 0}
avg_probs = {0: 0., 1: 0., 2: 0., 3: 0.}
avg_amps = {0: 0., 1: 0., 2: 0., 3: 0.}
max_amp = 0

max_freq = 0
it = 0           # current reading flag
f = f_base       # local frequency
EOCF = 0         # End of changing frequency flag
signal_arr = []

weak_avg_ctr = 0
weak_ctr = 0
avg_confidence = 0
strong_confidence_threshold = 0.6
weak_confidence_threshold = 0.85
current_pin = 0

vals = []
############################### support functions
##############################

def calc_running_amp(sig):
    global amp_slice_size
    running_amp = np.average(np.sort(np.abs(sig).flatten())[::-1][:amp_slice_size])
    #print(running_amp)
    return running_amp

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_signal_distance(signal: np.array) -> float:
    """
    Computes a peak to peak distanse of the input signal.

    Arguments:
        signal, np.array - an array of complex points of the input signal

    Returns:
        sig_dist, float - peak to peak amplitude signal distance
    """
    # sig_dist = np.max(np.abs(signal)) - np.min(np.abs(signal))
    sig_dist = np.abs(np.max(signal) - np.min(signal))
    return sig_dist


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
					"freq": 2400,
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

    global signal_tag

    global reading_signal_delay
    global iterations

    global max_amp
    global max_freq
    global it

    global f

    global EOCF
    global strong_model
    global weak_model

    global weak_ctr
    global weak_avg_ctr
    global avg_confidence
    global strong_confidence_threshold
    global weak_confidence_threshold
    global weak_samples_confidence
    global height_threshold
    global signal_arr
    global ctrs
    global avg_probs
    global avg_amps
    global amp_border
    global amp_decays
    global classes
    global pred_amps
    global weak_detected
    global vals
    outputs = []

    y = np.array(lvl).ravel()
    signal_arr = np.concatenate((signal_arr, y), axis=None)

    if f <= f_roof:
        f = f_base
        signal_arr = []
        send_data(np.max(np.array(vals)))
        vals = []
        return f, EOCF
    else:
        label = None
        if len(signal_arr) >= RK3588_MODEL['split_size']:  # the signal length `soft` constraint
            
            sig = np.array([signal_arr.real[0:RK3588_MODEL['split_size']], signal_arr.imag[0:RK3588_MODEL['split_size']]], dtype=np.float32)
            running_amp = calc_running_amp(sig)
            # feeds the input signal into weak classifier
            outputs = RK3588_MODEL['model'].inference(inputs=[sig])
            signal_arr = []

            label = np.argmax(outputs, axis=2)[0][0]
            probability = softmax(outputs[0][0])[label]

            #print(classes[label], round(probability, 2), int(f))

            weak_ctr += 1
            if (label != 0) and (label != 3) and probability > weak_confidence_threshold:
                weak_avg_ctr += 1
            ctrs[label] += 1
            avg_probs[label] += probability
            avg_amps[label] += running_amp
        if weak_ctr == RK3588_MODEL['N_predictions']:
            prob = round(softmax(outputs[0][0])[label], 2)
            #print('Detected: ', classes[label], ' Probability: ' , prob, 'Frequency: ', str(f))
            print('---------> Frequency: ', f)
            for key in avg_probs.keys():
                avg_probs[key] = float(avg_probs[key] / max(1, ctrs[key]))
                avg_amps[key] = float(avg_amps[key] / max(1, ctrs[key]))

            for key in ctrs.keys():
                print("avg prob: ", "%.4f" % avg_probs[key],"avg_amp: ", "%.4f" % avg_amps[key],"ctr: ", ctrs[key],"class: ",  classes[key])
            #sfm = softmax(list(avg_probs.values()))
            #print(sfm)
            label = np.argmax(list(ctrs.values()))
            weak_avg_ctr = ctrs[label]
            weak_avg_prob = avg_probs[label] # / max(1, ctrs[label])
            weak_avg_amp = avg_amps[label]
            amp_decay = 0 #amp_decays[min(np.sum(np.where(amp_border <= weak_avg_amp, 1, 0)), len(amp_border) - 1)]
            final_confidence_threshold = weak_confidence_threshold + amp_decay
            print('-' * 50, classes[label], weak_avg_ctr,"%.4f" % float(weak_avg_prob), '/', "%.4f" % final_confidence_threshold)
            #print(classes[label] , ': ', sfm[label])
            avg_probs = {0: 0., 1: 0., 2: 0., 3: 0.}
            avg_amps = {0: 0., 1: 0., 2: 0., 3: 0.}
            ctrs = {0: 0, 1: 0, 2: 0, 3: 0}
            if weak_avg_ctr >= RK3588_MODEL['N_predictions'] * RK3588_MODEL['N_samples_confidence_threshold'] and label != 0 and weak_avg_prob > final_confidence_threshold:
                print('!!!' * 30, f)
                vals.append(weak_avg_amp)
            else:
                vals.append(0)
            f += f_step

            weak_ctr = 0
            weak_avg_ctr = 0
        return f, EOCF

import time, os, argparse
import serial
#from termplt import Plot
import numpy as np
import re

port = "/dev/ttyS4"
#port = "/dev/ttyUSB0"
serport = serial.Serial(port, 115200, timeout=1)
#serport.open()
i = 1

import time, os, argparse

#from PIL import Image, ImageDraw
#from luma_examples.examples.demo_opts import get_device
#import serial

import wiringpi as wpi
from wiringpi import GPIO

import requests
import json
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


wpi.wiringPiSetup()


pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 7]
freqs = [
 5705,5725, 5732,
 5733,5740,5745,5752,5760,5765,5769,5771,
 5780,5785,5790,5800,5805,5806,5809,5820,
 5825,5828,5840,5843,5845,5847,5860,5865,
 5866,5880,5885,5905,
]
# setup LED pins

#for pin in pins:
#    wpi.pinMode(pin, GPIO.OUTPUT)
#    wpi.digitalWrite(pin, GPIO.LOW)


on_state = 0
off_state = 1

p2p_border = np.array([114, 119, 124, 129, 134, 139, 144, 149, 165, 175]) # 3.3V
#p2p_border = np.array([155,162, 170, 179, 189, 200, 212, 225, 239, 254]) # 5V v2
#p2p_border = np.array([]) # 5V v3

#amp_border = np.array([0, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39])


def calc_rssi_metrics(rssi, n_average=None):
    if n_average is not None:
        arr = list(zip(rssi, freqs))
        rssi = sorted(rssi, reverse=True)
        arr = sorted(arr, key=lambda row: row[0])
        print('Top freqs: ', end='')

        for i in range(0, n_average):
            print(arr[i][1], end= ' ')

        res = np.sum(np.array(rssi)[:n_average])/n_average
        return res
    else:
        return np.max(np.array(rssi))


def send_data(sig_dist):
    len_threshold = int(os.getenv('len_threshold'))
    localhost = os.getenv('lochost')
    localport = os.getenv('locport')
    length = int(np.sum(np.where(p2p_border <= sig_dist, 1, 0)))

    if length >= len_threshold:
        trigger = True
    else:
        trigger = False

    data_to_send = {"freq": 5800,
                    "amplitude": 33,
                    "triggered": trigger,
                    "light_len": length
                }

    response = requests.post("http://{0}:{1}/process_data".format(localhost, localport), json=data_to_send)

    if response.status_code == 200:
        print("Данные успешно отправлены и приняты!")
    else:
        print("Ошибка при отправке данных: ", response.status_code)


def indicate_distance_lightning(sig_dist: float) -> None:
    """
    Sets lightning according the ditance between the reciever and a drone.
    """
    global on_state
    global off_state

    length = np.sum(np.where(p2p_border <= sig_dist, 1, 0))
    print('|', '#' * length, '-' * (len(p2p_border) - length), '|', sep='')
    if length > 0:
        for pin in pins[:length]:
            wpi.digitalWrite(pin, on_state)

    if length != len(p2p_border):
        for pin in pins[length:]:
            wpi.digitalWrite(pin, off_state)

iteration = 1
losses = 0
while True:
    try:
        rssi = []
        while(len(rssi)) < 5:
            try:
                rssi = serport.readline().decode().replace('\n', '').replace('\r', '')[:-1].split(' ')
                rssi = np.array([int(x) for x in rssi][::-1])
            except Exception as e:
                pass


        metrics = calc_rssi_metrics(rssi, n_average=3)
        #print('rssi: ', rssi)
        print('#' * 20)
        print('* Iteration ', iteration)
        print('* Bad data: ', losses/iteration, '%')
        iteration += 1

        print('* Metrics: ', metrics)
        print('Average: ', np.sum(rssi) / len(rssi))

        send_data(metrics)
        #indicate_distance_lightning(metrics)
    except Exception as e:
        print(str(e))
        print(".", end='')
        losses += 1

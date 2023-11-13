#! /usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import requests
import torch
import subprocess
import threading
import time
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import numpy as np
import matplotlib as plt
import io
import cv2
import json
from json import JSONEncoder

app = Flask(__name__)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env-template')
load_dotenv(dotenv_path)
path_to_NN = os.getenv('path_to_my_nn')
local_host = os.getenv('local_host')
local_port = os.getenv('local_port')
classes = {0: 'drone', 1: 'noise', 2: 'wifi'}
#server_ip = os.getenv('to_server_ip')
#server_port = os.getenv('to_server_port')

url_local_server = "http://{0}:{1}/receive_data".format(local_host, local_port)
pre_data = []
freqs = [2400]
data_queue = [None]*len(freqs)
#scheduler = BackgroundScheduler(daemon=True)
#scheduler.start()

def sig2pic(data, figsize=(16, 8), dpi=80):
    try:
        #with open(path_to_data + filename, 'rb') as file:
            #tmp = np.frombuffer(file.read(), dtype=np.complex64)
        signal = data
        fig1 = plt.figure(figsize=figsize)
        plt.axes(ylim=(-1, 1))
        sigr = signal.real
        sigi = signal.imag
        plt.plot(sigr, color='black')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=dpi)
        buf1.seek(0)
        img_arr1 = np.frombuffer(buf1.getvalue(), dtype=np.uint8)
        buf1.close()
        img1 = cv2.imdecode(img_arr1, 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        plt.close()

        fig2 = plt.figure(figsize=figsize)
        plt.axes(ylim=(-1, 1))
        sigr = signal.real
        sigi = signal.imag
        plt.plot(sigi, color='black')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.close()
        img = np.array([img1, img2])
        return img
    except Exception as e:
        print(str(e))
        return None

'''
def register_module():
    try:
        url = "http://{0}:{1}/module/register/{2}".format(server_ip, server_port, mac_address)
        response = requests.post(url)
        response.raise_for_status()  # Проверка успешности запроса
        print("Модуль зарегистрирован успешно = ", mac_address)
    except requests.exceptions.RequestException as e:
        flag = 1
        print("Ошибка при регистрации модуля:" + str(e), mac_address)
'''

'''def get_mac_address(interface='eth0'):
    try:
        result = os.popen('sudo ifconfig ' + interface).read()
        mac_index = result.find('ether')  # Индекс начала строки с MAC-адресом
        if mac_index != -1:
            mac_address = result[mac_index + 6:mac_index + 23]
            return mac_address
        else:
            return None
    except Exception as e:
        print("Ошибка при получении MAC-адреса:" + str(e))
        return None
'''

'''
@scheduler.scheduled_job(IntervalTrigger(seconds=20))
def heartbeat():
    try:
        url = "http://{0}:{1}/module/heartbeat/{2}".format(master_server_ip, master_server_port, mac_address)
        response = requests.get(url)
        if response.status_code == 200:
            print("heartbeat")
        elif response.status_code == 404:
            print('heartbeat не был отправлен из-за отсутствия регистрации модуля')
            print('Повторим попытку регистрации')
            register_module()
        else:
            print('heartbeat не был отправлен по какой-то причине')
    except Exception:
        print('heartbeat не был отправлен из-за отстутствия сервера в поле видимости')
'''

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route('/receive_data', methods = ['POST'])
def receive_data():
    print('Перед Вайл')
    data = json.loads(requests.json)
    freq = int(data['freq'])
    img = np.asarray(sig2pic(np.asarray(data['data'])), dtype=np.float32)
    outputs = model(img.to(device))
    print('TOKEN' + str(int(data['token'])))
    print('OUTPUTS:')
    print(outputs)
    label = np.argmax(outputs, axis=2)[0][0]
    print('LABEL: ' + str(label))
    probability = softmax(outputs[0][0])[label]
    print(classes[label], round(probability, 2), int(freq))
    result = {'message': 'Data inference successfully'}
    return jsonify(result)

'''
def send_to(ModuleDataSingleV2, flag):
    try:
        if flag == 0:
            response = requests.post("http://{0}:{1}/data/single/{2}".format(master_server_ip, master_server_port, mac_address), json=ModuleDataSingleV2)
        else:
            response = requests.post("http://{0}:{1}/data/bulk/{2}".format(master_server_ip, master_server_port, mac_address), json=bulk_data)
            print('данные отправлены через балк')


        if response.status_code == 200:
            print('Данные успешно отправлены')
            flag = 0
            bulk_data.clear()
        elif response.status_code == 404:
            flag = 1
            if len(bulk_data) > max_len_bulk: # Если лимит bulk_data превышен, то удаляем первый элемент списка
                bulk_data.pop(0)
            bulk_data.append(ModuleDataSingleV2)
            print('Данные не были отправлены по причине отсутствия регистрации модуля')
            print('Будет выполнена еще одна попытка регистрации модуля')
            register_module()  # Регистрируем модуль
    except Exception:
        if len(bulk_data) > max_len_bulk: # Если лимит bulk_data превышен, то удаляем первый элемент списка
            bulk_data.pop(0)
        bulk_data.append(ModuleDataSingleV2)
        flag = 1
        print('Данные не были отправлены по причине отсутствия сервера в поле видимости')


@app.route('/process_data', methods = ['GET'])
def process_data():
    data = requests.get(url_local_server)
    if flow == 0:

        print('Received data: ', data)
        freq = data["freq"]
        #Агрегируем  N пакетов данных от частот в один общий список, он используется в функции agregate data. Каждая позиция списка фиксируется за отдельной частотой
        for i in range(len(freqs)):
            if freq == freqs[i]:
                data_queue[i] = data
            else:
                continue

        print('После получения данных data_queue выглядит следующим образом: ', data_queue)
        result = {'message' : 'Data processed successfully'}
        return jsonify(result)
    else:
        result = {"message" : "Data processed successfully"}
        return jsonify(result)

mac_address = get_mac_address()'''

if __name__ == '__main__':
	#register_module() # Регистрация модуля на сервере
	#update_gps_coordinates()
	#child = threading.Thread(target=add_data)  # Запуск агрегатора данных и отправки на мастер-сервер.
	#child.daemon = True
	#child.start()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('D:/InferencePCServer/NN/model.pt', map_location = device)
    model.eval()
    #print(model)
    '''with open('D:\InferencePCServer\A5_2.4_noise_0.npy', 'rb') as f:
        data = np.frombuffer(f.read(), np.complex64)
        img = torch.tensor([data.real, data.imag])
        outputs = model(img.to(device))
        print('OUTPUTS:')
        print(outputs)
        label = np.argmax(outputs, axis=2)[0][0]
        print('LABEL: ' + str(label))
        probability = softmax(outputs[0][0])[label]
        print(classes[label], round(probability, 2))'''
    app.run(host=local_host, port=local_port)  # Запуск сервера

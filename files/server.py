#! /usr/bin/env python
# -*- coding: utf-8 -*-
from importlib import import_module

import mlconfig
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import torch
import subprocess
import threading
import time
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
import json
from json import JSONEncoder
plt.switch_backend('agg')
app = Flask(__name__)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env-template')
load_dotenv(dotenv_path)
path_to_NN = os.getenv('path_to_my_nn')
example = os.getenv('example')
local_host = os.getenv('local_host')
local_port = os.getenv('local_port')
classes = {0: 'drone', 1: 'noise', 2: 'wifi'}
#server_ip = os.getenv('to_server_ip')
#server_port = os.getenv('to_server_port')

#url_local_server = "http://{0}:{1}/receive_data".format(local_host, local_port)
pre_data = []
freqs = [2400]
data_queue = [None]*len(freqs)
token = 1
#scheduler = BackgroundScheduler(daemon=True)
#scheduler.start()

def sig2pic(data_real, data_imag, figsize=(16, 8), dpi=80):
    try:
        #with open(path_to_data + filename, 'rb') as file:
            #tmp = np.frombuffer(file.read(), dtype=np.complex64)
        fig1 = plt.figure(figsize=figsize)
        plt.axes(ylim=(-1, 1))
        sigr = data_real
        sigi = data_imag
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
        sigr = data_real
        sigi = data_imag
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
    global model
    global token
    data = json.loads(request.json)
    print('Получен пакет ' + str(token))
    token += 1
    freq = int(data['freq'])
    img = np.asarray(sig2pic(np.asarray(data['data_real'], dtype=np.float32), np.asarray(data['data_imag'], dtype=np.float32)), dtype=np.float32)
    img = torch.tensor(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output.data, 1)
    prediction = np.asarray((pred.cpu()))[0]
    print('PREDICTION ' + str(classes[int(prediction)]))
    return str(classes[int(prediction)])


    '''print('OUTPUTS:')
    print(output)
    label = np.argmax(output, axis=2)[0][0]
    print('LABEL: ' + str(label))
    probability = softmax(output[0][0])[label]
    print(classes[label], round(probability, 2), int(freq))
    result = {'message': 'Data inference successfully'}
    return jsonify(result)


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


def load_function(attr):
    module_, func = attr.rsplit('.', maxsplit=1)
    return getattr(import_module(module_), func)


if __name__ == '__main__':
    config = mlconfig.load('config_resnet18.yaml')
    model = load_function(config.model.architecture)(pretrained=False)
    lin = model.conv1
    new_lin = torch.nn.Sequential(
        torch.nn.Conv2d(2, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        lin
    )
    model.conv1 = new_lin
    model.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        model = model.to(device)
    model.load_state_dict(torch.load("tree.pth",  map_location=device))
    model.eval()
    app.run(host='192.168.1.90', port='8080')
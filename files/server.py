#! /usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import requests
import subprocess
import threading
import time
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import json
import wiringpi as wpi
from wiringpi import GPIO
global flow
flow = 0
app = Flask(__name__)
wpi.wiringPiSetup()

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

local_host = os.getenv('lochost')
local_port = os.getenv('locport')
master_server_ip = os.getenv('master_server_ip')
master_server_port = os.getenv('master_server_port')
pins = json.loads(os.getenv('pins'))
pin_jammer = int(os.getenv('pin_jammer'))
len_threshold = int(os.getenv('len_threshold'))
time_to_jam = int(os.getenv('time_to_jam'))
time_to_fresh = int(os.getenv('time_to_fresh'))
latitude = float(os.getenv('latitude'))
longitude = float(os.getenv('longitude'))
active_interval_to_send = int(os.getenv('active_interval_to_send'))
passive_interval_to_send = int(os.getenv('passive_interval_to_send'))


flag = 0

max_len_bulk = 50
bulk_data = []

freqs = [1200, 2400, 5800]
data_queue = [None]*len(freqs)
# Создание планировщика
scheduler = BackgroundScheduler(daemon=True)
scheduler.start()

def wpi_on_boot():
	for pin in pins:
		wpi.pinMode(pin, GPIO.OUTPUT)
		wpi.digitalWrite(pin, GPIO.LOW)

	time.sleep(3)

	for pin in pins:
		wpi.digitalWrite(pin, GPIO.HIGH)

	wpi.pinMode(pin_jammer, GPIO.OUTPUT)
	wpi.digitalWrite(pin_jammer, GPIO.LOW)
	print('wpi_on_boot')


def register_module():
    try:
        url = "http://{0}:{1}/module/register/{2}".format(master_server_ip, master_server_port, mac_address)
        response = requests.post(url)
        response.raise_for_status()  # Проверка успешности запроса
        print("Модуль зарегистрирован успешно = ", mac_address)
    except requests.exceptions.RequestException as e:
        flag = 1
        print("Ошибка при регистрации модуля:" + str(e), mac_address)


def get_mac_address(interface='eth0'):
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

#@app.route('/get_gps', methods = ['POST'])
@scheduler.scheduled_job(IntervalTrigger(minutes=1))
def update_gps_coordinates():
    #data_gps = request.json
    result = {
        'latitude': latitude,
        'longitude': longitude
    }
    try:
        url = "http://{0}:{1}/data/gps/{2}".format(master_server_ip, master_server_port, mac_address)
        response = requests.post(url, json=result)
        if response.status_code == 200:
            print('gps успешно отправлен')
        elif response.status_code == 404:
            print('gps не был отправлен из-за отсутствия регистрации модуля на сервере')
            print('Будет выполнена еще одна регистрация')
            register_module()  # Регистрируем модуль
        else:
            print('gps не был отправлен по какой-то причине')

    except Exception:
        print('gps не были отправлены из-за отстутствия сервера в поле видимости')
    return result

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

def light(light_lens):
    global flow
   # flow = 0
    print("Количество светодиодов получены: ", light_lens)
    length = max(light_lens.values())
    if length > 0:
        for pin in pins[:length]:
            wpi.digitalWrite(pin, GPIO.LOW)
        if length >= len_threshold:
            print('Включили глушилку!!!')
            flow = 1
            wpi.digitalWrite(pin_jammer, GPIO.HIGH)
            for pin in pins:
                wpi.digitalWrite(pin, GPIO.HIGH)
            wpi.digitalWrite(pins[len(pins) - 2], GPIO.LOW)
            wpi.digitalWrite(pins[len(pins) - 1], GPIO.LOW)
            time.sleep(time_to_jam)
            # Глушилка включилась. Теперь нужно включить задержку скрипта модуля, причем
            # таким образом, чтобы сама глушилка отработала N секунд, а скрипт модуля
            # не принимал данные N+M секунду, где N - примерно пара минут, а M - примерно
            # 10 секунд. Нужно это для того, чтобы за 10 секунд после глушилки скрипты
            # отсканировали чистую местность и привели данные, которые они отправляют, в порядок
            # В качестве N и M взять time_to_jam и time_to_fresh из .env
            # (они уже загружены в этот скрипт).

            print('Время вышло. Выключили глушилку!!!')
            wpi.digitalWrite(pin_jammer, GPIO.LOW)
            wpi.digitalWrite(pins[len(pins) - 2], GPIO.HIGH)
            wpi.digitalWrite(pins[len(pins) - 1], GPIO.HIGH)
            time.sleep(time_to_fresh)
            flow = 0


    if length != 10:
        for pin in pins[length:]:
            wpi.digitalWrite(pin, GPIO.HIGH)


def agregate_data():
    print('Перед Вайл')
    while True:

        count = 0
        light_lens = {}
        data = []
        ModuleDataSingleV2 = {}

        if any(item is not None for item in data_queue):
            for item in data_queue:
                if item is not None:
                    light_lens[item["freq"]] = item.pop("light_len", None)
                    if item["triggered"]:
                         count += 1
                    data.append(item)

            now = datetime.utcnow()- timedelta(seconds=2)
            now = now.strftime("%m/%d/%Y %H:%M:%S")
            ModuleDataSingleV2 = {
                "registeredAt" : now,
                "data": data
            }
            print('На сервер-мастер будет отправлена следующая информация: ', ModuleDataSingleV2)
            send_to_master(ModuleDataSingleV2, flag)

            for i in range(len(freqs)):
                data_queue[i] = None

            print('После отправки data_queue выглядит так:', data_queue)
            light(light_lens)
            if count == 0:
                print('Следующие данные будут отправлены через {0} секунд'.format(passive_interval_to_send))
                for i in range(passive_interval_to_send):
                    drone = has_true = any(item is not None and item.get('triggered') is True for item in data_queue)
                    if drone:
                        break
                    time.sleep(1)
            else:
                print('1c')
                time.sleep(active_interval_to_send)

def send_to_master(ModuleDataSingleV2, flag):
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


@app.route('/process_data', methods = ['POST'])
def process_data():
    data = request.json
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

mac_address = get_mac_address()

if __name__ == '__main__':
	wpi_on_boot() # Настройка пинов в правильные положения. вкл и выкл светодиодов на секунду.
	register_module() # Регистрация модуля на сервере
	update_gps_coordinates()
	child = threading.Thread(target=agregate_data)  # Запуск агрегатора данных и отправки на мастер-сервер.
	child.daemon = True
	child.start()

	app.run(host=local_host, port=local_port)  # Запуск сервера

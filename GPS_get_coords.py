#! /usr/bin/env python
# -*- coding: utf-8 -*-
import serial
import time
import string
import pynmea2
import io
import re
import os
from dotenv import load_dotenv
# <VTG(true_track=None, true_track_sym='', mag_track=None, mag_track_sym='', spd_over_grnd_kts=None, spd_over_grnd_kts_sym='', spd_over_grnd_kmph=None, spd_over_grnd_kmph_sym='', faa_mode='N')>

class GPSReader():
    def __init__(self, port=os.getenv('gpsport'), baudrate=9600, timeout=0.3):
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._sio = io.TextIOWrapper(io.BufferedRWPair(self._ser,self._ser))
        self.gps_data = dict({'lat':0, 'lng':0})
        self.dataout = pynmea2.NMEAStreamReader()
    def get_coords(self):
        pass

    def get_gps_data(self):
        try:
            newdata=self._sio.readline()
            if newdata[0:6] == "$GPRMC":
                newmsg = pynmea2.parse(newdata)
                lat = newmsg.latitude
                lng = newmsg.longitude
                self.gps_data['lat'] = lat
                self.gps_data['lng'] = lng
            '''
            newdata = pynmea2.parse(newdata)
            k = repr(newdata)
            print(k)
            print(len(k))
            #lat = newdata.latitude
            #lng = newdata.longitude
            #sats = newdata.num_sats
            #self.gps_data["lat"] = lat
            #self.gps_data["lng"] = lng
            #self.gps_data["sats"] = sats
            if k[0] == '<':
                d = dict({})
                data = re.sub(r'[<()>,]', "", k)
                station = k[0:2]
                data = data[3:].split(' ')
                for i in data:
                    u = i.split('=')
                    d[u[0]] = u[1]
                self.gps_data[station] = d
                #print("!!!!!!!!!!!!!!!!!!!!!")
                #print(self.gps_data[station])
            '''
            '''
            if lat != 0 and lng != 0 and sats != 0:
                print('-' * 60)
                print("Latitude: " + str(lat) + " Longitude: " + str(lng) +  " num_sats: " + str(sats))                print('-' * 60)
                sats = 0
                lat = 0
                lng = 0
            '''
        except Exception as e:
            print(str(e))

load_dotenv('envserver.env')
gps = GPSReader()
localhost = os.getenv('lochost')
localport = os.getenv('locport')


while True:
    try:
        gps.get_gps_data()
    except Exception as e:
        print("1", str(e))

try:
    response = requests.post("http://{0}:{1}/get_gps".format(localhost, localport), json=gps)

    if response.status_code == 200:
        print("Данные успешно отправлены и приняты!")
    else:
        print("Ошибка при отправке данных:", response.status_code)
except Exception:
    print('gps не отправлен')
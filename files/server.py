from flask import Flask, request, jsonify
from importlib import import_module
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from Model import Model
import torch.nn as nn
import numpy as np
import mlconfig
import torch
import json
import cv2
import sys
import os
import io
import re

app = Flask(__name__)
plt.switch_backend('agg')
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
num_token = int(os.getenv('NUM_TOKEN'))
src_result = os.getenv('SRC_RESULT')

path_to_nn_1 = os.getenv('PATH_TO_NN_1')
path_to_nn_2 = os.getenv('PATH_TO_NN_2')
path_to_nn_3 = os.getenv('PATH_TO_NN_3')
src_ml_config_18 = os.getenv('SRC_ML_CONFIG_18')
src_ml_config_50 = os.getenv('SRC_ML_CONFIG_50')

model_list = []
classes = {0: 'drone', 1: 'noise', 2: 'wifi'}
count_test = 1
token = 0


def pre_func_conv(data=None):
    try:
        figsize = (16, 8)
        dpi = 80

        fig1 = plt.figure(figsize=figsize)
        plt.axes(ylim=(-1, 1))
        sig_real = data[0]
        plt.plot(sig_real, color='black')
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
        sig_imag = data[1]
        plt.plot(sig_imag, color='black')
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
        img = np.asarray([img1, img2], dtype=np.float32)

        print('Подготовка данных завершена')
        print()
        return img

    except Exception as e:
        print(str(e))
        return None


def build_func_conv_18(file_model='', file_config='', num_classes=None):
    try:
        config = mlconfig.load(file_config)
        model = getattr(import_module(config.model.architecture.rsplit('.', maxsplit=1)[0]),
                        config.model.architecture.rsplit('.', maxsplit=1)[1])()
        model.conv1 = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(7, 7), stride=(2, 2),
                                                          padding=(3, 3), bias=False), model.conv1)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cpu':
            model = model.to(device)
        model.load_state_dict(torch.load(file_model, map_location=device))
        model.eval()

        print('Инициализация модели завершена')
        print()
        return model

    except Exception as exc:
        print(str(exc))
        return None


def build_func_conv_50(file_model='', file_config='', num_classes=None):
    try:
        config = mlconfig.load(file_config)
        model = getattr(import_module(config.model.architecture.rsplit('.', maxsplit=1)[0]),
                        config.model.architecture.rsplit('.', maxsplit=1)[1])()
        model.conv1 = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(7, 7), stride=(2, 2),
                                                          padding=(3, 3), bias=False), model.conv1)
        model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Linear(in_features=128, out_features=num_classes, bias=True)
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cpu':
            model = model.to(device)
        model.load_state_dict(torch.load(file_model, map_location=device))
        model.eval()

        print('Инициализация модели завершена')
        print()
        return model

    except Exception as exc:
        print(str(exc))
        return None


def inference_func_conv(data=None, model=None, mapping=None, type_model='', model_id=0):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = torch.unsqueeze(torch.tensor(data), 0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predict = torch.max(output.data, 1)
        prediction = np.asarray(predict.cpu())[0]
        print('PREDICTION ' + str(model_id) + ' с типом ' + str(type_model) + ': ' + mapping[int(prediction)])

        label = np.asarray(np.argmax(output, axis=1))[0]
        output = np.asarray(torch.squeeze(output, 0))
        expon = np.exp(output - np.max(output))
        probability = (expon / expon.sum())[label]
        print('Уверенность модели ' + str(model_id) + ' с типом ' + str(type_model) + ' в предсказании: ' + str(
            round(probability, 3)))

        print('Инференс завершен')
        print()
        return prediction

    except Exception as exc:
        print(str(exc))
        return None


def post_func_conv(src='', model_type='', model_id=0, ind_inference=0, data=None, prediction=None, mapping=None):
    try:
        fig, ax = plt.subplots()
        ax.imshow(data[0], cmap='gray')
        plt.savefig(src + 'inference' + str(mapping[int(prediction)]) + '_real_' + str(model_id) + '_' + str(model_type) + '.png')
        fig, ax = plt.subplots()
        ax.imshow(data[1], cmap='gray')
        plt.savefig(src + str(mapping[int(prediction)]) + '_imag_' + str(model_id) + '_' + str(model_type) + '.png')
        plt.close()

        print('Постобработка завершена')
        print()

    except Exception as exc:
        print(str(exc))
        return None


def run_example(num_test=0):
    try:
        list_data = []
        list_prediction = []
        src_example = os.getenv('SRC_EXAMPLE')
        for _ in range(num_test):
            with open(src_example, 'r+') as data_file:
                list_data.append(np.frombuffer(data_file.read(), dtype=np.float32))
            try:
                for key in classes.keys():
                    if classes[key] in re.split('[._/]', src_example):
                        list_prediction.append(int(key))
            except Exception as exc:
                print(str(exc))

        for model in model_list:
            model.test_inference(list_data=list_data, list_prediction=list_prediction)

    except Exception as exc:
        print(str(exc))


@app.route('/receive_data', methods=['POST'])
def receive_data():
    global token

    print()
    data = json.loads(request.json)
    print('#' * 100)
    print('Получен пакет ' + str(token+1))
    freq = int(data['freq'])
    print('Частота ' + str(freq))

    for model in model_list:
        print('-' * 100)
        print(str(model))
        model.inference([np.asarray(data['data_real'], dtype=np.float32), np.asarray(data['data_imag'], dtype=np.float32)])
        print('-' * 100)

    token += 1
    print()
    print('#' * 100)

    if token == num_token:
        print('Завершение работы!')
        sys.exit()

    result_msg = {'message': 'Data inference successfully'}
    return jsonify(result_msg)


if __name__ == '__main__':

    init_model = Model(file_model=path_to_nn_1, file_config=src_ml_config_18, src_result=src_result, type_model='resnet_18_with_8',
                  build_model_func=build_func_conv_18, pre_func=pre_func_conv, inference_func=inference_func_conv,
                  post_func=post_func_conv, classes=classes)
    model_list.append(init_model)

    init_model = Model(file_model=path_to_nn_1, file_config=src_ml_config_18, src_result=src_result, type_model='resnet_18_without_8',
                  build_model_func=build_func_conv_18, pre_func=pre_func_conv, inference_func=inference_func_conv,
                  post_func=post_func_conv, classes=classes)
    model_list.append(init_model)

    init_model = Model(file_model=path_to_nn_3, file_config=src_ml_config_50, src_result=src_result, type_model='resnet_50',
                  build_model_func=build_func_conv_50, pre_func=pre_func_conv, inference_func=inference_func_conv,
                  post_func=post_func_conv, classes=classes)
    model_list.append(init_model)

    run_example(num_test=count_test)

    server_ip = os.getenv('SERVER_IP')
    server_port = os.getenv('SERVER_PORT')
    app.run(host=server_ip, port=server_port)

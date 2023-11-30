import time
from importlib import import_module
import mlconfig
from flask import Flask, request, jsonify
import torch
import os
import sys
from dotenv import load_dotenv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import cv2
import json

plt.switch_backend('agg')
app = Flask(__name__)
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
file_name_result = os.getenv('file_name_result')
num_token = int(os.getenv('num_token'))
src_result = os.getenv('src_result')


classes = {0: 'drone', 1: 'noise', 2: 'wifi'}
token = 1
result = ''
result_dict = {0: 0, 1: 0, 2: 0}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sig2pic(data_real, data_imag, figsize=(16, 8), dpi=80):
    try:
        fig1 = plt.figure(figsize=figsize)
        plt.axes(ylim=(-1, 1))
        sig_real = data_real
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
        sig_imag = data_imag
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
        img = np.array([img1, img2])
        return img

    except Exception as e:
        print(str(e))
        return None


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@app.route('/receive_data', methods=['POST'])
def receive_data():
    global model
    global token
    global result
    global result_dict

    data = json.loads(request.json)
    print()
    print('Получен пакет ' + str(token))

    freq = int(data['freq'])
    img = np.asarray(sig2pic(np.asarray(data['data_real'], dtype=np.float32), np.asarray(data['data_imag'], dtype=np.float32)), dtype=np.float32)
    img_torch = torch.unsqueeze(torch.tensor(img), 0).to(device)
    with torch.no_grad():
        output = model(img_torch)
        _, predict = torch.max(output.data, 1)
    prediction = np.asarray(predict.cpu())[0]
    result += str(int(prediction)) + ' '
    result_dict[int(prediction)] += 1
    print('PREDICTION ' + str(classes[int(prediction)]))
    label = np.asarray(np.argmax(output, axis=1))[0]
    output = np.asarray(torch.squeeze(output, 0))
    probability = softmax(output)[label]
    print('Уверенность модели в предсказании: ' + str(round(probability, 3)))
    print('Загрузка картинок ...')
    np.save(str(src_result) + 'result_' + str(token) + '_' + str(classes[int(prediction)]) + '.npy', img)
    fig, ax = plt.subplots()
    ax.imshow(img[0], cmap='gray')
    plt.savefig(src_result + 'token_' + str(token) + '_' + str(classes[int(prediction)]) + '_real.png')
    fig, ax = plt.subplots()
    ax.imshow(img[1], cmap='gray')
    plt.savefig(src_result + 'token_' + str(token) + '_' + str(classes[int(prediction)]) + '_imag.png')
    plt.close()
    token += 1
    print('Загрузка картинок завершена')
    print('Инференс завершён')
    print()

    if token-1 == num_token:
        print('Завершение работы')
        print(result_dict)
        with open(file_name_result, 'w') as f:
            f.write(result)
        print('Файл с результатами записан!')
        sys.exit()

    result_msg = {'message': 'Data inference successfully', 'prediction': classes[int(prediction)]}
    return jsonify(result_msg)


def load_function(attr):
    module_, func = attr.rsplit('.', maxsplit=1)
    return getattr(import_module(module_), func)


def build_model():
    path_to_nn = os.getenv('path_to_nn')
    src_ml_config = os.getenv('src_ml_config')

    config = mlconfig.load(src_ml_config)
    model = load_function(config.model.architecture)(pretrained=False)
    lin = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False), model.conv1)
    model.conv1 = lin
    model.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)

    if device != 'cpu':
        model = model.to(device)
    model.load_state_dict(torch.load(path_to_nn, map_location=device))
    model.eval()

    print('Модель загружена')
    print()
    return model


def run_example():
    print('Пробный запуск модели')
    src_example = os.getenv('src_example')

    img = torch.unsqueeze(torch.tensor(np.asarray(np.load(src_example, 'r+'), dtype=np.float32)), 0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predict = torch.max(output.data, 1)
    prediction = np.asarray(predict.cpu())[0]
    print('Пробное предсказание: ' + str(classes[int(prediction)]))
    label = np.asarray(np.argmax(output, axis=1))[0]
    output = np.asarray(torch.squeeze(output, 0))
    probability = softmax(output)[label]
    print('Уверенность модели в пробном предсказании: ' + str(round(probability, 3)))
    print('Тестовый инференс завершён')
    print()


if __name__ == '__main__':

    model = build_model()
    run_example()

    server_ip = os.getenv('server_ip')
    server_port = os.getenv('server_port')
    app.run(host=server_ip, port=server_port)
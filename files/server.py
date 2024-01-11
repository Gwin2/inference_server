from flask import Flask, request, jsonify
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from files.Model import Model
import numpy as np
import importlib
import json
import sys


app = Flask(__name__)
plt.switch_backend('agg')

token = 0
model_list = []

config = dict(dotenv_values(".env"))
num_token = int(config['NUM_TOKEN'])


def init_model_list():
    global model_list

    try:
        for key in config.keys():
            if key.startswith('NN_'):
                params = config[key].split(' && ')
                module = importlib.import_module('files.Models.' + params[4])
                classes = {}
                for value in params[9][1:-1].split(','):
                    classes[len(classes)] = value
                model_list.append(Model(file_model=params[0], file_config=params[1], src_example=params[2],
                                        src_result=params[3], type_model=params[4],
                                        build_model_func=getattr(module, params[5]),
                                        pre_func=getattr(module, params[6]), inference_func=getattr(module, params[7]),
                                        post_func=getattr(module, params[8]),
                                        classes=classes))
    except Exception as exc:
        print(str(exc))


def run_example():
    try:
        for model in model_list:
            model.test_inference()
    except Exception as exc:
        print(str(exc))


@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
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
            Model.get_result_list()
            print('\nЗавершение работы!')
            sys.exit()

        result_msg = {'message': 'Data inference successfully!'}
        return jsonify(result_msg)

    except Exception as exc:
        print(str(exc))


if __name__ == '__main__':
    init_model_list()
    run_example()
    app.run(host=config['SERVER_IP'], port=config['SERVER_PORT'])

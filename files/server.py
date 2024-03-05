from flask import Flask, request, jsonify
from files.Algorithm import Algorithm
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from files.Model import Model
import numpy as np
import importlib
import shutil
import json
import sys
import os


app = Flask(__name__)
plt.switch_backend('agg')


alg_list = []
model_list = []
config = dict(dotenv_values(".env"))
num_token_nn = int(config['NUM_TOKEN_NN'])
num_token_alg = int(config['NUM_TOKEN_ALG'])


def init_data_for_inference():
    try:
        if os.path.isdir(config['SRC_RESULT']):
            shutil.rmtree(config['SRC_RESULT'])
        os.mkdir(config['SRC_RESULT'])
        if os.path.isdir(config['SRC_EXAMPLE']):
            shutil.rmtree(config['SRC_EXAMPLE'])
        os.mkdir(config['SRC_EXAMPLE'])
    except Exception as exc:
        print(str(exc))
        print()

    try:
        global model_list
        for key in config.keys():
            if key.startswith('NN_'):
                params = config[key].split(' && ')
                module = importlib.import_module('files.Models.' + params[4])
                classes = {}
                for value in params[9][1:-1].split(','):
                    classes[len(classes)] = value
                model = Model(file_model=params[0], file_config=params[1], src_example=params[2], src_result=params[3],
                              type_model=params[4], build_model_func=getattr(module, params[5]),
                              pre_func=getattr(module, params[6]), inference_func=getattr(module, params[7]),
                              post_func=getattr(module, params[8]), classes=classes, number_synthetic_examples=int(params[10]),
                              number_src_data_for_one_synthetic_example=int(params[11]), path_to_src_dataset=params[12])
                model_list.append(model)
            if key.startswith('ALG_'):
                params = config[key].split(' && ')
                module = importlib.import_module('files.Algorithms.' + params[2])
                classes = {}
                for value in params[6][1:-1].split(','):
                    classes[len(classes)] = value
                alg = Algorithm(src_example=params[0], src_result=params[1], type_alg=params[2], pre_func=getattr(module, params[3]),
                                inference_func=getattr(module, params[4]), post_func=getattr(module, params[5]), classes=classes,
                                number_synthetic_examples=int(params[7]), number_src_data_for_one_synthetic_example=int(params[8]), path_to_src_dataset=params[9])
                alg_list.append(alg)
    except Exception as exc:
        print(str(exc))
        print()


def run_example():
    try:
        for model in model_list:
            model.get_test_inference()
    except Exception as exc:
        print(str(exc))


@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        print()
        data = json.loads(request.json)
        print('#' * 100)
        print('Получен пакет ' + str(Model.get_ind_inference()))
        freq = int(data['freq'])
        print('Частота ' + str(freq))

        for model in model_list:
            print('-' * 100)
            print(str(model))
            model.get_inference([np.asarray(data['data_real'], dtype=np.float32), np.asarray(data['data_imag'], dtype=np.float32)])
            print('-' * 100)
            print()

        Model.get_inc_ind_inference()
        print()
        print('#' * 100)

        if Model.get_ind_inference() == num_token_nn + 1:
            Model.get_result_list()
            print('\nЗавершение работы!')
            sys.exit()

        for alg in alg_list:
            print('-' * 100)
            print(str(alg))
            alg.get_inference([np.asarray(data['data_real'], dtype=np.float32), np.asarray(data['data_imag'], dtype=np.float32)])
            print('-' * 100)
            print()

        Algorithm.get_inc_ind_inference()
        print()
        print('#' * 100)

        if Algorithm.get_ind_inference() == num_token_alg + 1:
            Algorithm.get_result_list()
            print('\nЗавершение работы!')
            sys.exit()

        result_msg = {'message': 'Data inference successfully!'}
        return jsonify(result_msg)

    except Exception as exc:
        print(str(exc))


if __name__ == '__main__':
    init_data_for_inference()
    # run_example()
    app.run(host=config['SERVER_IP'], port=config['SERVER_PORT'])

import numpy as np
import os
import re


class Model(object):
    _model_id = 0
    _inference_list = {}

    @staticmethod
    def get_model_id():
        Model._model_id += 1
        return Model._model_id

    @staticmethod
    def append_to_inference_list(type_model='', ind_inference=0, prediction='', probability=0.0):
        Model._inference_list[type_model].append(list([ind_inference, prediction, probability]))

    @staticmethod
    def get_inference_list(type_model=''):
        return Model._inference_list[type_model]

    def __init__(self, file_model='', file_config='', src_example='', src_result='', type_model='',
                 build_model_func=None, pre_func=None, inference_func=None, post_func=None, classes=None):
        Model._inference_list[type_model] = []
        self._model_id = Model.get_model_id()
        self._file_model = file_model
        self._file_config = file_config
        self._src_example = src_example
        self._src_result = src_result
        self._type_model = type_model
        self._build_model_func = build_model_func
        self._pre_func = pre_func
        self._inference_func = inference_func
        self._post_func = post_func
        self._classes = classes
        self._ind_inference = 0
        self._data = None
        self._shablon = ' Модель ' + str(self._model_id) + ' с типом ' + str(self._type_model)
        self._model = self._build_model()

    def __str__(self):
        return self._shablon + ' работает!' + '\n'

    def _build_model(self):
        print('Инициализация' + self._shablon)
        return self._build_model_func(file_model=self._file_model, file_config=self._file_config,
                                      num_classes=len(self._classes))

    def _prepare_data(self, data=None):
        print('Подготовка данных' + self._shablon)
        self._data = self._pre_func(data)

    def _post_data(self, prediction=''):
        print('Постобработка данных' + self._shablon)
        self._ind_inference += 1
        self._post_func(src=self._src_result, data=self._data, model_id=self._model_id,
                        ind_inference=self._ind_inference, prediction=prediction)

    def test_inference(self):
        try:
            count_access = 1
            count_attempt = 1
            _, _, files = next(os.walk(self._src_example))
            print(files)
            for file in files:
                self._src_example += file
                with open(self._src_example, 'r+') as data_file:
                    self._data = np.frombuffer(data_file.read(), dtype=np.float32)

                print()
                self._prepare_data(data=self._data)

                print('Тестовый инференс' + self._shablon + ' попытка ' + str(count_attempt))
                prediction = self._inference_func(data=self._data, model=self._model, mapping=self._classes,
                                                  shablon=self._shablon)

                for key in self._classes.keys():
                    if self._classes[key] in re.split('[._/]', self._src_example):
                        if int(key) == prediction:
                            print('Тест ' + str(count_attempt) + ' пройден!')
                            count_access += 1
                        else:
                            print('Тест ' + str(count_attempt) + ' провален!')
                        count_attempt += 1
                        break

            print('Тестовый инференс' + self._shablon + ' пройден с результатом ' + str(100*(count_access-1)/(count_attempt-1) + ' %'))
            print()

        except Exception as exc:
            print(str(exc))

    def inference(self, data=None):
        self._prepare_data(data=data)
        print('Инференс' + self._shablon)
        prediction, probability = self._inference_func(data=self._data, model=self._model, mapping=self._classes,
                                                       shablon=self._shablon)
        Model.append_to_inference_list(type_model=self._type_model, ind_inference=self._ind_inference,
                                       prediction=prediction, probability=probability)
        self._post_data(prediction=prediction)

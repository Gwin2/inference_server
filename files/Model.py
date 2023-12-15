import torch
import numpy as np


class Model(object):
    _model_id = 0

    def __init__(self, file_model='', file_config='', build_model_func=None, pre_func=None, inference_func=None,
                 post_func=None, mapping=None):
        self._data = None
        self._model_id = Model.get_model_id()
        self._file_model = file_model
        self._file_config = file_config
        self._build_model_func = build_model_func
        self._pre_func = pre_func
        self._inference_func = inference_func
        self._post_func = post_func
        self._mapping = mapping
        self._model = self._build_model()

    @staticmethod
    def get_model_id():
        Model._model_id += 1
        return Model._model_id

    def _build_model(self):
        print('Инициализация модели ' + str(self._model_id))
        return self._build_model_func(self._file_model, self._file_config)

    def _get_prepared_data(self, pre_func, data):
        print('Подготовка данных модели ' + str(self._model_id))
        self._data = pre_func(data)

    @classmethod
    def _softmax(cls, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _inference(self):
        self._get_prepared_data(self._pre_func, self._data)
        print('Инференс модели ' + str(self._model_id))
        with torch.no_grad():
            output = self._model(self._data)
            _, predict = torch.max(output.data, 1)
        prediction = np.asarray(predict.cpu())[0]
        print('PREDICTION ' + str(self._model_id) + ' ' + self._mapping[int(prediction)])
        label = np.asarray(np.argmax(output, axis=1))[0]
        output = np.asarray(torch.squeeze(output, 0))
        probability = self._softmax(output)[label]
        print('Уверенность модели ' + str(self._model_id) + ' в предсказании: ' + str(round(probability, 3)))
        self._get_post_data()

    def _get_post_data(self):
        print('Постобработка данных модели ' + str(self._model_id))
        self._post_func(self._data)

    def __str__(self):
        return 'Модель ' + str(self._model_id) + ' работает!'

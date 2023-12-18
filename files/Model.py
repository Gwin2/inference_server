class Model(object):
    _model_id = 0

    @staticmethod
    def get_model_id():
        Model._model_id += 1
        return Model._model_id

    def __init__(self, file_model='', file_config='', src_result='', type_model='', build_model_func=None,
                 pre_func=None, inference_func=None, post_func=None, classes=None):
        self._model_id = Model.get_model_id()
        self._file_model = file_model
        self._file_config = file_config
        self._src_result = src_result
        self._type_model = type_model
        self._build_model_func = build_model_func
        self._pre_func = pre_func
        self._inference_func = inference_func
        self._post_func = post_func
        self._classes = classes
        self._ind_inference = 0
        self._data = None
        self._model = self._build_model()

    def __str__(self):
        return 'Модель ' + str(self._model_id) + ' с типом ' + str(self._type_model) + ' работает!' + '\n'

    def _build_model(self):
        print('Инициализация модели ' + str(self._model_id) + ' с типом ' + str(self._type_model))
        return self._build_model_func(file_model=self._file_model, file_config=self._file_config,
                                      num_classes=len(self._classes))

    def _prepare_data(self, data=None):
        print('Подготовка данных модели ' + str(self._model_id) + ' с типом ' + str(self._type_model))
        self._data = self._pre_func(data)

    def _post_data(self, prediction=None):
        print('Постобработка данных модели ' + str(self._model_id) + ' с типом ' + str(self._type_model))
        self._post_func(src=self._src_result, data=self._data, prediction=prediction, mapping=self._classes,
                        model_id=self._model_id)

    def test_inference(self, list_data=None, list_prediction=None):
        try:
            ind_attempt = 0
            access = 0
            for data in list_data:
                self._prepare_data(data=data)

                print()
                print('Тестовый инференс модели ' + str(self._model_id) + ' с типом ' + str(self._type_model) + ' попытка ' + str(ind_attempt+1))
                prediction = self._inference_func(data=self._data, model=self._model, mapping=self._classes,
                                              type_model=self._type_model, model_id=self._model_id)

                if list_prediction[ind_attempt] == prediction:
                    print('Тест ' + str(ind_attempt+1) + ' пройден')
                    access += 1
                else:
                    print('Тест ' + str(ind_attempt+1) + ' провален')
                ind_attempt += 1

            print('Тестовый инференс модели ' + str(self._model_id) + ' с типом ' + str(self._type_model) + ' пройден с результатом ' + str(100*access/ind_attempt) + ' %')
            print()

        except Exception as exc:
            print(str(exc))

    def inference(self, data=None):
        self._prepare_data(data=data)
        print('Инференс модели ' + str(self._model_id) + ' с типом ' + str(self._type_model))
        prediction = self._inference_func(data=self._data, model=self._model, mapping=self._classes,
                                          type_model=self._type_model, model_id=self._model_id)
        self._post_data(prediction=prediction)

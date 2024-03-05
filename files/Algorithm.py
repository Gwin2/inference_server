from tqdm import tqdm
import numpy as np
import random
import os
import re


class Algorithm(object):
    _alg_id = 0
    _ind_inference = 1
    _result_list = dict()

    @staticmethod
    def get_alg_id():
        try:
            return Algorithm._alg_id
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def _get_inc_alg_id():
        try:
            Algorithm._alg_id += 1
            return Algorithm._alg_id
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def get_ind_inference():
        try:
            return Algorithm._ind_inference
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def get_inc_ind_inference():
        try:
            Algorithm._ind_inference += 1
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def _init_result_list(type_alg=''):
        try:
            Algorithm._result_list[type_alg] = {}
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def get_result_list():
        try:
            def get_max(dict_inf=None):
                try:
                    return max([(len(i[0]) if len(i) else 0) for i in dict_inf.values()])
                except Exception as error:
                    print(str(error))
                    return 0

            max_length_label = max([get_max(i) for i in Algorithm._result_list.values()])
            num_inf = max(list(map(int, list(Algorithm._result_list.values())[0].keys())))
            max_length_type_alg = max([len(i) for i in Algorithm._result_list.keys()])
            num_gaps = (max_length_type_alg + 4) * (len(Algorithm._result_list) + 1) + 2 * (len(Algorithm._result_list) + 2)
            print('_' * num_gaps)

            print('||' + ' ' * (max_length_type_alg + 4) + '|', end='')
            for type_alg in Algorithm._result_list.keys():
                print('|' + ' ' * ((max_length_type_alg - len(type_alg)) // 2 + 2), end='')
                print(type_alg, end='')
                print(' ' * ((max_length_type_alg - len(type_alg)) // 2 + 2) + '|', end='')
            print('|')

            for ind_inf in range(1, num_inf+1):
                print('||' + ' ' * ((max_length_type_alg - len(str(ind_inf))) // 2 + 2), end='')
                print(str(ind_inf) if len(str(ind_inf)) % 2 == 0 else str(ind_inf) + ' ', end='')
                print(' ' * ((max_length_type_alg - len(str(ind_inf))) // 2 + 2) + '|', end='')

                for info_inference in Algorithm._result_list.values():
                    if len(info_inference[ind_inf]) != 0:
                        length_gap_left = (max_length_label - len(info_inference[ind_inf][0])) // 2 + (max_length_label - len(info_inference[ind_inf][0])) % 2
                        length_gap_right = (max_length_label - len(info_inference[ind_inf][0])) // 2 + 1
                        to_print = (' ' * length_gap_left + info_inference[ind_inf][0] + ' ' * length_gap_right) + (str(info_inference[ind_inf][1])
                                                                    if len(str(info_inference[ind_inf][1])) != 3 else str(info_inference[ind_inf][1]) + ' ')
                    else:
                        to_print = ' ' * max_length_type_alg
                    print('|' + ' ' * ((max_length_type_alg - len(to_print)) // 2 + 2), end='')
                    print(to_print, end='')
                    print(' ' * ((max_length_type_alg - len(to_print)) // 2 + 2) + '|', end='')
                print('|')

            print('_' * num_gaps)
        except Exception as exc:
            print(str(exc))

    @staticmethod
    def _add_in_result_list(type_alg='', ind_inference=0, list_to_add=None):
        try:
            Algorithm._result_list[type_alg][ind_inference] = list_to_add
        except Exception as exc:
            print(str(exc))

    def __init__(self, src_example='', src_result='', type_alg='', pre_func=None, inference_func=None, post_func=None,
                 classes=None, number_synthetic_examples=0, number_src_data_for_one_synthetic_example=0, path_to_src_dataset=''):
        try:
            self._p=0
            self._src_example = src_example
            self._src_result = src_result
            self._type_alg = type_alg
            self._pre_func = pre_func
            self._inference_func = inference_func
            self._post_func = post_func
            self._classes = classes
            self._num_outputs = len(self._classes.keys())
            self._number_synthetic_examples = number_synthetic_examples
            self._number_src_data_for_one_synthetic_example = number_src_data_for_one_synthetic_example
            self._path_to_src_dataset = path_to_src_dataset
            self._data = None
            self._shablon = ' Алгоритм ' + str(self._alg_id+1) + ' с типом ' + str(self._type_alg)
            self._alg_id = Algorithm._get_inc_alg_id()
            self._init_result_list(type_alg=self._type_alg)
        except Exception as exc:
            print(str(exc))

    def __str__(self):
        try:
            return self._shablon + ' работает!' + '\n'
        except Exception as exc:
            print(str(exc))

    def get_shablon(self):
        return self._shablon

    def _prepare_data(self, data=None):
        try:
            if not self._data:
                print('Подготовка данных' + self._shablon)
                self._data = []
            self._data.append(data)
            if len(self._data) == 10:
                self._data = self._pre_func(self._data)
                self._p=1
        except Exception as exc:
            print(str(exc))

    def _post_data(self, prediction=None):
        print('Постобработка данных' + self._shablon)
        self._ind_inference += 1
        self._post_func(prediction=prediction)

    def get_test_inference(self):
        try:
            self._test_inference()
        except Exception as exc:
            print(str(exc))

    def _create_synthetic_examples(self):
        try:
            print('#' * 100)
            print('Создание синтетических примеров: ' + self._shablon)

            path_to_example_directory = os.path.join(self._src_example, self._type_alg)
            os.mkdir(path_to_example_directory)
            for ind in tqdm(range(self._number_synthetic_examples)):
                try:
                    label = self._classes[random.randint(0, self._num_outputs-1)]
                    path_to_src_directory = os.path.join(self._path_to_src_dataset, label)
                    with open(path_to_src_directory + '/' + os.listdir(path_to_src_directory)[0], 'rb') as data_file:
                        data = np.frombuffer(data_file.read(), dtype=np.float32)
                    array_example = np.zeros(np.shape(data))
                    for _ in range(self._number_src_data_for_one_synthetic_example):
                        with open(path_to_src_directory + '/' + random.choice(os.listdir(path_to_src_directory)), 'rb') as data_file:
                            data = np.frombuffer(data_file.read(), dtype=np.float32)
                        array_example = np.add(array_example, data)
                    np.save(path_to_example_directory + '/' + label + '_' + str(ind+1), array_example / self._number_src_data_for_one_synthetic_example)
                except Exception as exc:
                    print(str(exc))

            print('Создание синтетических примеров завершено!')
            print()
        except Exception as exc:
            print(str(exc))
            print()

    def _test_inference(self):
        try:
            self._create_synthetic_examples()

            count_access = 1
            count_attempt = 1
            path_to_example = os.path.join(self._src_example, self._type_alg)
            _, _, files = next(os.walk(path_to_example))

            if files:
                ind_inference = 0
                for file in files:
                    with open(path_to_example + '/' + file, 'rb') as data_file:
                        self._data = np.frombuffer(data_file.read(), dtype=np.float32)

                    print()
                    self._prepare_data(data=self._data)

                    print('Тестовый инференс' + self._shablon + ' попытка ' + str(count_attempt))
                    prediction = self._inference_func(data=self._data, mapping=self._classes, shablon=self._shablon)

                    for value in self._classes.values():
                        if value in re.split('[._/]', file):
                            if value == prediction:
                                print('Тест ' + str(count_attempt) + ' пройден!\n')
                                count_access += 1
                            else:
                                print('Тест ' + str(count_attempt) + ' провален!\n')
                            count_attempt += 1
                            break

                    print()
                    print('Постобработка данных' + self._shablon)
                    ind_inference += 1
                    self._post_func(src=path_to_example+'/', data=self._data, ind_inference=ind_inference, alg_id=self._alg_id, alg_type=self._type_alg, prediction=prediction)

                print('\nТестовый инференс' + self._shablon + ' пройден с результатом ' + str(100 * (count_access - 1) / (count_attempt - 1)) + ' %')
                print('#' * 100)
                print()
            else:
                print('\nНет данных для тестового инференса')
                print()

        except Exception as exc:
            print(str(exc))

    def get_inference(self, data=None):
        try:
            self._inference(data=data)
        except Exception as exc:
            print(str(exc))

    def _inference(self, data=None):
        try:
            Algorithm._add_in_result_list(type_alg=self._type_alg, ind_inference=self.get_ind_inference(), list_to_add=[])
            self._prepare_data(data=data)
            if self._p == 1:
                print('Подготовка данных завершена')
                print()
                print('Инференс' + self._shablon)
                prediction, metrica = self._inference_func(data=self._data)
                Algorithm._add_in_result_list(type_alg=self._type_alg, ind_inference=self.get_ind_inference(), list_to_add=[prediction, metrica])
                self._post_data(prediction=prediction)
                self._data = None
                self._p=0
        except Exception as exc:
            print(str(exc))

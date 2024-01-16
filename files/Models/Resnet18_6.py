from importlib import import_module
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import mlconfig
import torch
import cv2
import io


def build_func_resnet18(file_model='', file_config='', num_classes=None):
    try:
        config = mlconfig.load(file_config)
        model = getattr(import_module(config.model.architecture.rsplit('.', maxsplit=1)[0]),
                        config.model.architecture.rsplit('.', maxsplit=1)[1])()
        model.conv1 = torch.nn.Sequential(torch.nn.Conv2d(2, 3, kernel_size=(7, 7), stride=(2, 2),
                                                          padding=(3, 3), bias=False), model.conv1)
        model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=128, bias=True),
                                 nn.Linear(in_features=128, out_features=32, bias=True),
                                 nn.Linear(in_features=32, out_features=3, bias=True))

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


def pre_func_resnet18(data=None):
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
        print(img.shape)
        return img

    except Exception as e:
        print(str(e))
        return None


def inference_func_resnet18(data=None, model=None, mapping=None, shablon=''):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img = torch.unsqueeze(torch.tensor(data), 0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predict = torch.max(output.data, 1)
        prediction = mapping[int(np.asarray(predict.cpu())[0])]
        print('PREDICTION' + shablon + ': ' + prediction)

        label = np.asarray(np.argmax(output, axis=1))[0]
        output = np.asarray(torch.squeeze(output, 0))
        expon = np.exp(output - np.max(output))
        probability = round((expon / expon.sum())[label], 2)
        print('Уверенность' + shablon + ' в предсказании: ' + str(probability))

        print('Инференс завершен')
        print()
        return prediction, probability

    except Exception as exc:
        print(str(exc))
        return None


def post_func_resnet18(src='', model_type='', prediction='', model_id=0, ind_inference=0, data=None):
    try:
        fig, ax = plt.subplots()
        ax.imshow(data[0], cmap='gray')
        plt.savefig(src + '_inference_' + str(ind_inference) + '_' + prediction + '_real_' + str(model_id) + '_' + model_type + '.png')
        fig, ax = plt.subplots()
        ax.imshow(data[1], cmap='gray')
        plt.savefig(src + '_inference_' + str(ind_inference) + '_' + prediction + '_imag_' + str(model_id) + '_' + model_type + '.png')
        plt.close()

        print('Постобработка завершена')
        print()

    except Exception as exc:
        print(str(exc))
        return None

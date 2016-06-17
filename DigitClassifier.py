# -*- coding:utf-8 -*-
from NeuralNetwork import NeuralNetwork
from InputCanvas import InputCanvas
from PIL import Image, ImageFilter
from numpy import random
import numpy as np
import tkinter

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})


class DigitClassifier(tkinter.Tk):
    def __init__(self):
        tkinter.Tk.__init__(self)
        self.nn = NeuralNetwork(784, 300, 10)

        self.background = tkinter.Canvas(self, width = 308, height = 308)
        self.background.config(background="black")
        self.input_canvas = InputCanvas(self, width = 300, height = 300)
        self.result_label = tkinter.Label(self, text='')
        self.recog_button = tkinter.Button(self, text='Recognize', command=self.recognize)
        self.clear_button = tkinter.Button(self, text='Clear', command=self.input_canvas.clear)

        self.background.pack()
        self.input_canvas.place(x=4, y=4)
        self.result_label.pack()
        self.recog_button.pack()
        self.clear_button.pack()

    def train_nn(self, epochs=100000, edit_image=False):
        """ニューラルネットワークを訓練する"""
        import Mnist
        labels = Mnist.trainLabels
        images = Mnist.trainImages
        inputs, targets = [], []
        for _ in range(epochs):
            i = int(random.random() * len(labels))
            target = np.zeros(10)
            if edit_image:
                # 訓練データを加工する
                img = Image.fromarray(images[i])
                new_img = Image.new('L', (28, 28))
                new_img.paste(img.rotate(random.uniform(-45.0, 45.0)),
                              (random.randint(-5.0, 5.0), random.randint(-5.0, 5.0)))
                image = np.asarray(new_img).ravel()
            else:
                # 加工なし
                image = images[i].ravel()
            inputs.append(image/255.0)
            target[labels[i]] = 1.0
            targets.append(target)
        print("start training...")
        self.nn.train(np.array(inputs), np.array(targets), n=0.01)

        labels = Mnist.testLabels
        images = Mnist.testImages
        inputs, targets = [], []
        for i in range(len(labels)):
            target = np.zeros(10)
            inputs.append(images[i].ravel() / 255.0)
            target[labels[i]] = 1.0
            targets.append(target)
        print("start testing...")
        results = self.nn.test(np.array(inputs), np.array(targets))
        #print(results)

        overall = np.zeros((10, 10), dtype=int)
        correct = 0
        for result, target in zip(results, targets):
            ri = max(enumerate(result), key=lambda x: x[1])[0]
            ti = max(enumerate(target), key=lambda x: x[1])[0]
            overall[ti, ri] += 1
            if ti == ri:
                correct += 1
        print(overall)
        print(float(correct)/len(labels))

        # 訓練後のパラメータを保存する
        np.save('parameters/w1_2.npy', self.nn.w1_2)
        np.save('parameters/w2_3.npy', self.nn.w2_3)

    def load_nn_parameters(self):
        # パラメータを読み込む
        self.nn.w1_2 = np.load('parameters/w1_2.npy')
        self.nn.w2_3 = np.load('parameters/w2_3.npy')

    def recognize(self):
        # キャンバスに書き込まれた数字を認識する
        img = self.input_canvas.getImage().filter(ImageFilter.BLUR).convert('L')
        img.thumbnail((28, 28), getattr(Image, 'ANTIALIAS'))
        img = img.point(lambda x: 255 - x)
        input = np.asarray(img).ravel()
        result = self.nn.test([input / 255.0], np.zeros(10))[0]
        num = max(enumerate(result), key=lambda x: x[1])[0]
        self.result_label.configure(text = str(num))
        print(num, result)


def main():
    root = DigitClassifier()
    root.load_nn_parameters()
    #root.train_nn(edit_image=False)
    root.mainloop()


if __name__ == '__main__':
    main()
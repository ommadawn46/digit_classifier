# -*- coding:utf-8 -*-
from PIL import Image
from urllib import request
import numpy as np
import struct
import gzip
import os


def _download_mnist():
    """mnistデータが存在しない場合，ダウンロードする"""
    mnist_paths = ['mnist/train-labels-idx1-ubyte.gz', 'mnist/train-images-idx3-ubyte.gz', 'mnist/t10k-labels-idx1-ubyte.gz', 'mnist/t10k-images-idx3-ubyte.gz']
    mnist_urls = ['http://yann.lecun.com/exdb/' + path for path in mnist_paths]
    for path, url in zip(mnist_paths, mnist_urls):
        if not os.path.exists('mnist'):
            os.mkdir('mnist')
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                print("downloading... " + url)
                f.write(request.urlopen(url).read())


def _load_labels(path):
    """ラベルデータの読み込み"""
    print('loading labels... ' + path)
    with gzip.open(path, 'rb') as f:
        mn, num = struct.unpack('>2i', f.read(8))
        return np.array(struct.unpack('>%dB'%num, f.read()), dtype=np.uint8)


def _load_images(path):
    """数字画像の読み込み"""
    print('loading images... ' + path)
    with gzip.open(path, 'rb') as f:
        mn, num, row, col = struct.unpack('>4i', f.read(16))
        return np.array([[struct.unpack('>%dB'%col, f.read(col)) for __ in [0]*row] for _ in [0]*num], dtype=np.uint8)

_download_mnist()
trainLabels = _load_labels('mnist/train-labels-idx1-ubyte.gz')
trainImages = _load_images('mnist/train-images-idx3-ubyte.gz')
testLabels = _load_labels('mnist/t10k-labels-idx1-ubyte.gz')
testImages = _load_images('mnist/t10k-images-idx3-ubyte.gz')

def showImage(n):
    """画像を表示する"""
    Image.fromarray(trainImages[n]).show()
    print(trainLabels[n])
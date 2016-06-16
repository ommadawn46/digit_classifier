# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import struct


def _loadLabels(path):
    with open(path, 'rb') as f:
        mn, num = struct.unpack('>2i', f.read(8))
        return np.array(struct.unpack('>%dB'%num, f.read()))


def _loadImages(path):
    with open(path, 'rb') as f:
        mn, num, row, col = struct.unpack('>4i', f.read(16))
        return np.array([struct.unpack('>%dB'%(row*col), f.read(row*col)) for _ in [0]*num])


trainLabels = _loadLabels('mnist/train-labels.idx1-ubyte')
trainImages = _loadImages('mnist/train-images.idx3-ubyte')
testLabels = _loadLabels('mnist/t10k-labels.idx1-ubyte')
testImages = _loadImages('mnist/t10k-images.idx3-ubyte')


def showImage(n):
    img = Image.new('L', (28, 28))
    pixs = img.load()
    i = 0
    for y in range(28):
        for x in range(28):
            pixs[x, y] = int(trainImages[n][i])
            i += 1
    img.show()
    print(trainLabels[n])

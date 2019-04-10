import network
import numpy as np
import cv2


def test():
    img = cv2.imread(f'BioID-FaceDatabase-V1.2/h_0358.pgm', -1)
    dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized = resized.reshape((1, 4332)) / 255
    print(net.feedForward(resized))

    img = cv2.imread(f'BioID-FaceDatabase-V1.2/f_0077.pgm', -1)
    dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized = resized.reshape((1, 4332)) / 255
    print(net.feedForward(resized))

scale_percent = 20
net = network.Network([4332, 5500, 1000, 100, 1])

images = []
output = []

for i in range(1521):
    i = str(i)
    while len(i) < 4:
        i = '0' + i
    
    try:
        img = cv2.imread(f'BioID-FaceDatabase-V1.2/h_{i}.pgm', -1)
        o = 1
        dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))
    except:
        img = cv2.imread(f'BioID-FaceDatabase-V1.2/f_{i}.pgm', -1)
        o = 0
        dim = (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100))

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized = resized.reshape((1, 4332)) 
    images.append(resized[0] / 255)
    output.append(o)

inputs = np.array(images)
outputs = np.array([output]).T

IMG = np.concatenate((inputs, outputs), axis=1)
np.random.shuffle(IMG)

if __name__ == '__main__':
    test()
    r = 30
    for i in range(0, 1500, r):
        net.train(IMG[i:i+r], r * 10, 0.01)
        test()

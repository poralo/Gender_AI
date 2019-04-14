import network
import numpy as np
import cv2

def load_data():
    scale_percent = 20

    images = []
    output = []

    for i in range(1000, 1521):
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

    return IMG

if __name__ == '__main__':
    data = load_data()
    net = network.Network([4332, 5500, 1000, 100, 1])
    net.load()

    somme = 0

    for img in data:
        res = net.feedForward(img[:-1])
        if res > 0.5:
            res = 1
        else:
            res = 0
        
        somme += abs(res - img[-1])
    
    print(somme / 521)
import network
import numpy as np
import cv2
from random import shuffle

def load_data(train_percentage, doc_name='BioID-FaceDatabase-V1.2'):
    """Permet la mise en forme des données"""
    # Pourcentage de réduction des images
    scale_percent = 80

    men_imgs = []
    women_imgs = []

    print("Chargement des 1521 images")
    for i in range(1521):

        i = str(i)
        while len(i) < 4:
            i = '0' + i
        
        # Réduction de la taille des images + transformation des matrices images en vecteurs images 
        try:
            img = cv2.imread(f'{doc_name}/h_{i}.pgm', -1)
            dim = (int(img.shape[1] * (100 - scale_percent) / 100), int(img.shape[0] * (100 - scale_percent) / 100))

            resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            x = resized_img.reshape((1, 4332))
            men_imgs.append(x[0] / 255)

        except:
            img = cv2.imread(f'BioID-FaceDatabase-V1.2/f_{i}.pgm', -1)
            dim = (int(img.shape[1] * (100 - scale_percent) / 100), int(img.shape[0] * (100 - scale_percent) / 100))

            resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            x = resized_img.reshape((1, 4332))
            women_imgs.append(x[0] / 255)

    # Choix du nombre d'images d'entrainement pour qu'il y est autant d'homme que de femme
    nb_train = int(len(women_imgs) * train_percentage)
    nb_test = len(women_imgs) - nb_train

    print(f"Il y a {len(women_imgs)} images de femme dont {nb_train} vont servir pour l'entrainement et {nb_test} serviont à tester le réseau.")
    print(f"Il y a {len(men_imgs)} images d'homme dont {nb_train} vont servir pour l'entrainement et {nb_test} serviront à tester le réseau.")
    print()

    # Mise en forme des données en rendant l'ordre aléatoire
    shuffle(women_imgs)
    shuffle(men_imgs)

    features_train = np.concatenate((women_imgs[:nb_train], men_imgs[:nb_train]))
    labels_train = np.concatenate(([np.zeros(nb_train)], [np.ones(nb_train)]), axis=1).T
    imgs_train = np.concatenate((features_train, labels_train), axis=1)
    np.random.shuffle(imgs_train)

    features_test = np.concatenate((women_imgs[nb_train:nb_test], men_imgs[nb_train:nb_test]))
    labels_test = np.concatenate(([np.zeros(nb_test)], [np.ones(nb_test)]), axis=1).T
    imgs_test = np.concatenate((features_train, labels_train), axis=1)
    np.random.shuffle(imgs_test)

    return (imgs_train[:, :-1], np.array([imgs_train[:, -1]]).T, imgs_test[:, :-1], np.array([imgs_test[:, -1]]).T)

if __name__ == '__main__':
    features_train, labels_train, features_test, labels_test = load_data(0.8)

    net = network.Network([4332, 5500, 1000, 1])
    # net.load()
    print("Précision avant l'entrainement : ", net.score(features_test, labels_test))
    net.train(features_train, labels_train, 100, 10, 0.1)
    print("Précision après l'entrainement : ", net.score(features_test, labels_test))
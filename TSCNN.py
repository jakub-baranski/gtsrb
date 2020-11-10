import os
from pathlib import Path

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, \
    Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import load_model


class TSCNN:

    @staticmethod
    def create(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        model.add(Conv2D(8, (5, 5), padding="same",
                         input_shape=input_shape))
        # Filters - Zacząć od niższych wartości. W kolejnych warstwach rozszerzać ( np x 2 ).
            # Każdy filtr stara się znaleźć feature w obrazku poprzez zmianę wartości. Może rozmazać, wyostrzyć, wykryć krawędzie
        # Kernel size - zawsze nieparzyste - od 1x1 do 7x7 zazwyczaj. Większe dla większych rozmiarów obrazków. Zazwyczaj symetryczne (ale nie jest to wymóg?)
            # Określa rozmiar "convolution window" na którym stosowane są filtry
        # Padding - same - nie zmniejsza rozmiarów obrazka. "valid" zmniejsza. Po co jest padding -
            # Jeżeli stride (o ile jednostek przesuwa się kernel) jest większy niż 1 to jest prawdopodobieństwo, że
            # nie zmieści się w obrazku. Padding pozwoli na uniknięcie tego.

        model.add(Activation("relu"))
        # rectified linear activation - zwraca wartości dodatnie, albo 0
        model.add(BatchNormalization(axis=chan_dim))
        # Normalizacja - zwiększa stabilność sieci. Re-centruje i Re-skaluje
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Zmniejsza wymiary "obrazka", zmniejszając ilość pikseli. Filtr o podanej wielkości porusza się
            # po obrazku, bierze max i zwraca to z tego obszaru. MaxPooling2D zmniejszy obrazek 4-krotnie
            # Z czasem zmniejsza ilość parametrów sieci i tym samym obciążenie obliczeniowe.
            # Zmniejsza overfitting

        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Flatten())
        # Spłaszcza macierz. Zmniejsza wymiary do 1, <liczba-elementów-w-tensorze>. Czyli np warstwę 3, 16 zmieni na 1, 48
        model.add(Dense(128))
        # Zwyczajna warstwa. Każdy neuron jest połączony z neuronem w kolejnej i poprzedniej warstwie
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Losowo wyłącz część połączeń między neuronami. Zapobiega overfittingowi.

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # Wektor prawdopodobieństwa kategorii

        return model

    def load_saved(self, date=None):
        base_path = './trained-model'
        if date:
            path = f'{base_path}/{date}'
        else:
            path = max(Path(base_path).glob('*'), key=os.path.getctime)
        return load_model(f'{path}')

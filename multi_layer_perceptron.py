import random

import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


class MultiLayerPerceptron:

    def __init__(self, features, hidden_size, output_size, input_size=None, data_div_ratio=0.7, learning_rate=0.5, train_set=None, test_set=None, random_seed=None):

        self.input_size = input_size if input_size else features.shape[1] - 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.features = features
        self.ratio = data_div_ratio
        self.learning_rate = learning_rate

        if random_seed:
            np.random.seed(random_seed)

        random.shuffle(self.features)
        self.train_set = self.features[:int(
            len(self.features) * self.ratio)] if train_set is None else train_set
        self.test_set = self.features[int(
            len(self.features) * self.ratio):] if test_set is None else test_set

        w1 = np.random.rand(self.hidden_size,
                            self.input_size) * 0.01
        b1 = np.random.rand(self.hidden_size)

        w2 = np.random.rand(self.output_size, self.hidden_size) * 0.01
        b2 = np.random.rand(self.output_size)

        self.relevant_data = {
            "w1": w1, "b1": b1, "w2": w2, "b2": b2
        }

    def leaky_relu_activation(self, x, der=False):
        """Esta es una funcion de activacion.

        Leaky RELU resuelve el problema del RELU moribundo (cuando los valores de entrada son negativos, 
        los valores de activación son cero, por lo que algunos nodos mueren y no se actualizan durante la retropropagación) y 
        reduce la tendencia al problema del desvanecimiento de gradiente, 
        pero sigue existiendo la posibilidad de que se produzcan que los gradientes 
        pueden llegar a ser demasiado grandes durante la retroprogagación, 
        especialmente en redes neuronales de gran tamaño.

        Parametros:
            x float[]: Lista con los features (estimulos) que se quiere clasficar
            der boolean: Si es la derivada de la funcion.

        Return:
            data float[]: Los estimulos sometidos a la funcion de activacion

        """

        if der:
            data = [1 if item > 0 else 0.05 for item in x]
        else:
            data = [max(0.05 * item, item) for item in x]

        return np.array(data, dtype=float)

    def sigmoid_act(self, x, der=False):
        """Esta es una funcion de activacion.

        Esta función toma cualquier valor real como entrada y 
        da como salida valores en el rango de 0 a 1. 
        Cuanto mayor sea la entrada (más positiva), 
        más cerca estará el valor de salida de 1,0, 
        mientras que cuanto menor sea la entrada (más negativa), 
        más cerca estará la salida de 0.0.

        Parametros:
            x float[]: Lista con los features (estimulos) que se quiere clasficar
            der boolean: Si es la derivada de la funcion.

        Return:
            data float[]: Los estimulos sometidos a la funcion de activacion

        """

        data = 1/(1 + np.exp(-x))

        if der:
            data = data * (1 - data)

        return data

    def softmax_act(self, x, der=False):

        """Esta es una funcion de activacion.

        Softmax es una función de activación que convierte números en probabilidades. 
        La salida de Softmax es un vector/lista/matriz con probabilidades de cada resultado posible.

        Parametros:
            x float[][]: Lista con los features (estimulos) que se quiere clasficar
            der boolean: Si es la derivada de la funcion.

        Return:
            data float[][]: Los estimulos sometidos a la funcion de activacion

        """

        if der:
            data = x.reshape(-1, 1)
            return np.diagflat(data) - np.dot(data, data.T)
        else:
            data = np.exp(x)
            return data / np.sum(data)

    def forward(self, x, y):
        """La fase forward, en la que las activaciones se propagan de la capa de entrada a la de salida.

        Lo que ocurre en esta capa es el paso de la capa de entrada, a la oculta (u ocultas, en caso de ser mas de una)
        secuencialmente, hasta llegar a la capa de salida y realizar una primera prediccion.

        Parametros:
            x float[]: Lista con los features (estimulos) que se quiere clasficar
            y float[]: Lista con el label esperado para esos estimulos.

        Return: 
            z1 float[]: El resultado en la transicion de la capa de entrada a la oculta.
            a1 float[]: El resultado z1 pasando por la funcion de activacion.
            z2 float[]: El resultado en la transicion de la capa oculta a la de salida.
            a2 float[]: El resultado z2 pasando por la funcion de activacion. Softmax si la salida espera mas de una fila.
        
        """

        z1 = np.dot(self.relevant_data["w1"], x) + self.relevant_data["b1"]
        a1 = self.leaky_relu_activation(z1)

        z2 = np.dot(self.relevant_data["w2"],
                    a1) + self.relevant_data["b2"]
        if (y.shape[0] == 1):
            a2 = self.sigmoid_act(z2)
        else:
            a2 = self.softmax_act(z2)

        self.relevant_data.update({"z1": z1, "a1": a1, "z2": z2, "a2": a2})

        return z1, a1, z2, a2

    def cost(self, y):

        """El calculo de costos (deltas)

        Se calcula el costo de cada transicion entre capa y capa usando los derivados de las funciones en la activacion.
        El comienzo de la retropropagacion

        Parametros:
            y float[]: Lista con el label esperado para esos estimulos.

        Return: 
            delta1: delta de la primera transicion (input - hidden).
            delta2: delta de la segunda transicion (hidden - output).
        
        """

        a1, a2 = self.relevant_data["a1"], self.relevant_data["a2"]

        delta2 = (a2 - y) * self.sigmoid_act(a2, der=True) if (
            y.shape[0] == 1) else (a2 - y) * self.softmax_act(a2, der=True)
        delta1 = np.dot(
            delta2, self.relevant_data['w2']) * self.leaky_relu_activation(a1, der=True)
        return delta1, delta2

    def gradient_descent(self, x, y):

        """Fase de retropropagacion usando gradiente descendente

        Usando la salida obtenida del forward, se hace un proceso hacia atras corrigiendo los pesos
        acorde a los costos obtenidos.

        Parametros:
            x float[]: Lista con los features (estimulos) que se quiere clasficar
            y float[]: Lista con el label esperado para esos estimulos.

        """

        delta1, delta2 = self.cost(y)

        self.relevant_data["w2"] = self.relevant_data["w2"] - \
            self.learning_rate * delta2 * self.relevant_data["a1"]
        self.relevant_data["b2"] = self.relevant_data["b2"] - \
            self.learning_rate * delta2

        self.relevant_data["w1"] = self.relevant_data["w1"] - self.learning_rate * \
            np.kron(delta1, x).reshape(self.hidden_size, x.shape[0])
        self.relevant_data["b1"] = self.relevant_data["b1"] - \
            self.learning_rate * delta1

    def loss(self, y):
        """Calcula la pérdida entre las salidas de la fase forward y los objetivos.

        Parametros:
            y float[]: Lista con el label esperado para esos estimulos.

        Return:
            l float: valor de la perdida.

        """
        l = (self.relevant_data["a2"]-y)**2
        return l

    def training(self):
        """Proceso de entrenamiento dividido en tres fases principales.

        Por cada estimulo, pasamos primero por la fase de avance o propagacion hacia adelante (forward)
        desde el input hasta la capa del output.

        Luego, se hace la retropropagacion para actualizar y mejorar los pesos.

        Finalmente, se calcula la perdida.

        """
        loss_it = []
        i = 0
        for f in self.train_set:
            i += 1

            x = f[:len(f) - 1]
            y = f[len(f) - 1:]

            self.forward(x, y)
            self.gradient_descent(x, y)

            loss_it.append(self.loss(y))

        return loss_it

    def predict(self):
        """Proceso de prediccion.

        Por cada estimulo, pasamos primero por la fase de avance o propagacion hacia adelante (forward)
        desde el input hasta la capa del output usando los pesos obtenidos del entrenamiento. 

        Con un threshold de 0.5, se clasifica el resultado obtenido de la red.

        """
        predictions = []

        for f in self.test_set:
            x = f[:len(f) - 1]
            y = f[len(f) - 1:]

            self.forward(x, y)

            if self.relevant_data["z2"] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions


def main():
    earthSpace = pd.read_csv(
        "/home/mariangela/Downloads/EarthSpace.csv", header=None)
    medSci = pd.read_csv("/home/mariangela/Downloads/MedSci.csv",  header=None)

    earthSpace[len(earthSpace.columns)] = 0
    medSci[len(medSci.columns)] = 1

    data = np.concatenate((earthSpace.to_numpy(), medSci.to_numpy()))

    model = MultiLayerPerceptron(data, 15, 1, random_seed=42)

    model.training()

    predictions = model.predict()

    cm = metrics.confusion_matrix(
        [row[len(row) - 1:][0] for row in model.test_set], predictions)
    tp, tn, fp, fn = cm[0][0], cm[1][1], cm[1][0], cm[0][1]

    print("Accuracy: {acc} | Precission: {p} | Sensitivity: {sn} | Specificity: {sp}".format(
        acc=(tp + tn) / (tp + tn + fp + fn), p=tp / (tp + fp), sn=tp / (tp + fn), sp=tn / (tn + fp)))

    topics = {
        0: 'Earth Space',
        1: 'Med Sci'
    }

    df_cm = pd.DataFrame(cm, index=[topics[i] for i in range(0, 2)], columns=[
                         topics[i] for i in range(0, 2)])
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Reds, fmt='g')
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("Actual Label", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()

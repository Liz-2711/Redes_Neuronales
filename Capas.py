import json 
import numpy as np 
import matplotlib.pyplot as plt



def Load_Json(path_file):
    with open (path_file, "r") as file:
        
        data = json.load(file)

    print("opened")


    print(data.keys())


    print(data["input_shape"])
    print(data["preprocess"])
    print("Preprocess scale:", data['preprocess']['scale'])
    print(f"Número de capas: {len(data['layers'])}\n")
    this = data["layers"][0]
    print(data['layers'][0]['activation'])

    

    for i,layer in enumerate(data["layers"]):
        print(i, layer['units'])
        print(layer['activation'])

        W= np.array('W')
        b= np.array('b')

        print(f"Capa {i+1}:")
        print(f"  Tipo: {layer['type']}")
        print(f"  Unidades: {layer['units']}")
        print(f"  Activación: {layer['activation']}")
        print(f"  Forma W: {W.shape}, Forma b: {b.shape}")
    
    return data

    
modelo = Load_Json('Files/mnist_mlp.json')


def relu(z):

    return np.maximum(0, z)

def softmax(z):
   
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(z_stable)
    return exps / np.sum(exps, axis=1, keepdims=True)

class DenseLayer:
    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = activation
    
    def forward(self, x):
        output = np.dot(x, self.W) + self.b
        if self.activation == 'relu':
            return relu(output)
        elif self.activation == 'softmax':
            return softmax(output)
        #can be done with diccionario 

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.layers = []
        
        for capa in config['layers']:
            W = np.array(capa['W'])
            b = np.array(capa['b'])
            self.layers.append(DenseLayer(W, b, capa['activation']))
    
    def predict(self, X):
        output_pred = X
        for layer in self.layers:
            output_pred = layer.forward(output_pred)
        return output_pred


red = NeuralNetwork(modelo)


datos = np.load('Files/mnist_test.npz')
images = datos['images']
labels = datos['labels']


print(f"\nDimensiones originales:")
print(f"  images: {images.shape}")  
print(f"  labels: {labels.shape}")  

# Normalizar y aplanar
X = images.astype(np.float32) / modelo['preprocess']['scale']  # /255.0
X = X.reshape(X.shape[0], -1)  # (10000, 784)

print(f"Dimensiones después de preprocesamiento:")
print(f"  X: {X.shape}")  
print(f"  Rango: [{X.min():.3f}, {X.max():.3f}]")


# Forward pass
probabilidades = red.predict(X)
print(f"\nForma de la salida: {probabilidades.shape}")  # (10000, 10)


print(f"Suma de probabilidades (ejemplo 0): {probabilidades[0].sum():.6f}")

# Obtener clases predichas
predicciones = np.argmax(probabilidades, axis=1)


accuracy = np.mean(predicciones == labels)
print(f"\nExactitud del modelo: {accuracy*100:.2f}%")

casos_a_mostrar = [1200, 126, 274] 

print("\n--- DETALLE DE PROBABILIDADES DEL CASO 0 ---")
for caso in casos_a_mostrar:
    print(f"\n--- DETALLE DE PROBABILIDADES DE LA IMAGEN {caso} ---")
    print(f"Etiqueta Real: {labels[caso]} | Predicción: {predicciones[caso]}")
    for i, prob in enumerate(probabilidades[caso]):
        # Resaltamos la predicción con un asterisco
        marca = " <-- (Máxima)" if i == predicciones[caso] else ""
        print(f"  Dígito {i}: {prob:.4f}{marca}")


plt.figure(figsize=(15, 5))

for idx, caso in enumerate(casos_a_mostrar):
    img_original = images[caso].reshape(28, 28)
    etiqueta_real = labels[caso]
    clase_predicha = predicciones[caso]
    probs = probabilidades[caso]
    
    # Imagen
    plt.subplot(1, 3, idx + 1)
    plt.imshow(img_original, cmap='gray')
    plt.title(f"Real: {etiqueta_real} | Pred: {clase_predicha}\nSuma probs: {probs.sum():.4f}")
    plt.axis('off')

plt.tight_layout()
plt.show()


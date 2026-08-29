import json 
import numpy as np 




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


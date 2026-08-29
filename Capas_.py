import json 
import numpy as np 



def Load_Json():
    with open ("mnist_mlp_pretty.json", "r") as file:
        data = json.load(file)




    print(data.keys())


    print(data["input_shape"])
    print(data["preprocess"])
    this = data["layers"][0]
    print(data['layers'][0]['activation'])

    layers = []

    for i,layer in enumerate(data["layers"]):
        print(i, layer['units'])
        print(layer['activation'])

        W= np.array('W')
        b= np.array('b')

        print(f"  Forma W: {W.shape}, Forma b: {b.shape}")

        layers.append({
                'W': W,
                'b': b,
                'activation': layer['activation']
            })
    return data,layers

class Dense_Layer:
    def __inti__(self, weights, bias, activation):
        self.W=weights
        self.b=bias 
        self.activation = activation

        def forward(entrada):
            output = np.dot(entrada,self.W) + self.b


           
            if self.activation == 'relu':
                return np.maximum(0, output)
          
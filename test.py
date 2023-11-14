import numpy as np
import random
import pickle


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


sizes = [784, 30, 10]

biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)
           for x, y in zip(sizes[:-1], sizes[1:])]
with open('data/mind.pkl', 'wb') as memory:
    pickle.dump({'b': biases, 'w': weights}, memory)
print('Memória salva com sucesso')

'''
a = np.ones((5, 1))
print('viés: ', biases)
with open('matrizes_complexas.pkl', 'wb') as arquivo:
    pickle.dump({'b': biases, 'w': weights}, arquivo)

# Carregar as matrizes do arquivo
with open('matrizes_complexas.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Acessar as matrizes carregadas separadamente
matriz_carregada1 = dados_carregados['b']
matriz_carregada2 = dados_carregados['w']

print('mtrz: ', matriz_carregada1)
'''
'''
print(biases[1].transpose())
print('='*20)
print(biases[1])
print('='*20)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

'''
'''
def feedforward(x):
    
    for b, w in zip([1, 1, 1], (np.array([[1], [1]]), 1, 1)):
        print(f'Esse - {w}')
        a = (np.dot(w, a) + b)
        print(f'esse2 - {a}')
    return a
    '''
'''
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(biases, weights):
        print(w)
        z = np.dot(w, activation) + b
        print(z.shape)
        zs.append(z)
        print(z, 'zzzzzzz')
        activation = sigmoid(z)
        print('activation:', activation.shape)
        activations.append(activation)
    return activations

m = feedforward(a)
print(weights[-1],'aaaaaaaaaaaaaaaaaaaaaaaaa')
for n in m:
    print('n:', n)
'''
'''
a = [10]
print(a)
print(feedforward(a), "aaaaa")


print(np.argmax(a))
print(np.argmax(a, 0))
'''

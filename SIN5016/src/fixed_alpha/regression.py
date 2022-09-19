import pandas as pd
import numpy as np

def main():
    iris = pd.read_csv (r'../../data/iris.csv')
    
    X = iris.iloc[:, 0:4].values
    Y = iris.iloc[:, -1].values

    x_max = X.max(axis=0)
    x_min = X.min(axis=0)
    X = (X - x_min)/(x_max - x_min)

    Xt = np.hstack([X, np.ones((len(X), 1))])

    k = len(np.unique(Y))
    classes = list(np.unique(Y))

    Yt = np.zeros((len(Xt), k))
    for i, y in enumerate(Y):
        idx = classes.index(y)
        Yt[i][idx] = 1

    return regression(Xt, Yt, k)

def regression(Xt, Yt, k):
    _, d = Xt.shape

    alpha = 1e-5
    W = np.random.rand(k, d)

    S = softmax(np.matmul(Xt, W.T))
    erro = S - Yt
    grad = np.matmul(erro.T, Xt)
    norm = np.linalg.norm(grad.flatten())

    f = open("output_fixed_alpha.txt","w+")
    it = 0
    while norm > 1e-5 and it <= 1000:
        S = softmax(np.matmul(Xt, W.T))
        erro = S - Yt
        grad = np.matmul(erro.T, Xt)
        norm  = np.linalg.norm(grad.flatten())

        W = W - alpha * grad
        
        loss = cross_entropy(Yt, S)

        f.write(f"it: {it}, grad_norm: {norm}, cross_entropy: {loss}\n")
        it += 1

def softmax(S):
    expY = np.exp(S)
    return  expY / np.sum(expY, axis=1).reshape(S.shape[0], 1)

def cross_entropy(Yt, S):
    return -np.sum(np.multiply(Yt, np.log(S)))

main()
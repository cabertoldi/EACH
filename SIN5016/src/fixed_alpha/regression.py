import pandas as pd
import numpy as np

def main():
    iris = pd.read_csv (r'../../data/iris.csv')
    
    X = iris.iloc[:, 0:4].values
    Y = iris.iloc[:, -1].values
    Xt = np.hstack([X, np.ones((len(X), 1))])

    k = len(np.unique(Y))
    classes = list(np.unique(Y))

    Yt = np.zeros((len(Xt), k))
    for i, y in enumerate(Y):
        idx = classes.index(y)
        Yt[i][idx] = 1

    return regression(Xt, Yt, k)

def regression(Xt, Yt, k):
    N, d = Xt.shape

    alpha = 1e-5
    W = np.random.rand(k, d)

    S = softmax(np.matmul(Xt, W.T), N)
    erro = S - Yt
    grad = np.matmul(erro.T, Xt)
    loss_values = []

    grad_norm = np.linalg.norm(grad.flatten())

    f = open("output_fixed_alpha.txt","w+")
    it = 0
    while grad_norm > 1e-5 and it <= 1000:
        W = W - alpha * grad

        S = softmax(np.matmul(Xt, W.T), N)
        erro = S - Yt
        grad = np.matmul(erro.T, Xt)
        grad_norm  = np.linalg.norm(grad.flatten())

        loss = cross_entropy(Yt, S)
        loss_values.append(loss)

        it += 1
        f.write(f"it: {it}, grad_norm: {grad_norm}, cross_entropy: {loss}\n")

def softmax(S, N):
    expY = np.exp(S)
    return  expY / np.sum(expY, axis=1).reshape(N, 1)

def cross_entropy(Yt, S):
    return -np.sum(np.multiply(Yt, np.log(S)))

main()
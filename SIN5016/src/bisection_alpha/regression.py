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

    W = np.random.rand(k, d)
    S = softmax(np.matmul(Xt, W.T))
    erro = S - Yt
    grad = np.matmul(erro.T, Xt)
    grad_norm = np.linalg.norm(grad.flatten())

    idx = 0
    idx_max = 1000
    loss = 1e-5

    f = open("output_bisection_alpha.txt","w+")
    while grad_norm > loss and idx <= idx_max:
        idx += 1

        S = softmax(np.matmul(Xt, W.T))
        erro = S - Yt
        grad = np.matmul(erro.T, Xt)
        norm = np.linalg.norm(grad)
        grad_norm = np.divide(grad, norm)

        alpha = bisection(W, grad_norm, Xt, Yt)
        print("alpha", alpha)
        W = W - alpha * grad_norm

        loss = cross_entropy(Yt, S)

        f.write(f"it: {idx}, grad_norm: {norm}, cross_entropy: {loss}\n")


def h_l(alpha, W, grad, Xt, Yt):
    Wi = W - alpha * (-grad)
    S = softmax(np.matmul(Xt, Wi.T))
    
    erro = S - Yt
    grad_alpha = np.matmul(erro.T, Xt).flatten()

    result = np.dot(grad_alpha.T, -grad.flatten())
    print(result)
    return result

def h(alpha, W, grad, Xt, Yt):
    Wi = W - alpha * grad
    S = softmax(np.matmul(Xt, Wi.T))

    return cross_entropy(Yt, S)

def bisection(W, grad, Xt, Yt): 
    def alpha_gen():
        alpha_g = np.random.rand()
        while h_l(alpha_g, W, grad, Xt, Yt) < 0:
            alpha_g = np.random.rand()

        return alpha_g

    alpha_l = 0
    alpha_u = alpha_gen()
    print('alpha_u', alpha_u)
    alpha = (alpha_l + alpha_u) / 2

    hl = h_l(alpha, W, grad, Xt, Yt)
    print('hl', hl)

    it = 0
    it_max = np.int(np.ceil(np.log(alpha_u - alpha_l) - np.log(1e-5))/np.log(2))
    print('it_max', it_max)
    # while abs(hl) > 1e-5:
    while (it < it_max):
        it += 1
        if hl > 0:
            alpha_u = alpha
        elif hl < 0:
            alpha_l = alpha

        alpha = (alpha_l + alpha_u) / 2
        hl = h_l(alpha, W, grad, Xt, Yt)

    return alpha

def softmax(S):
    expY = np.exp(S)
    return  expY / np.sum(expY, axis=1).reshape(S.shape[0], 1)

def cross_entropy(Yt, S):
    return -np.sum(np.multiply(Yt, np.log(S)))

main()
import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    N = X.shape[0]
    D = X.shape[1]
    y = y.view(N,)

    w = torch.randn(D, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    for i in range(steps):
        # print("weight: ", w, b)
        
        out = X @ w + b
        # print(out)
        
        loss = torch.mean((out - y)**2)
        # print("Loss: ", loss)

        # print("Loss grad_fn:", loss.grad_fn)
        loss.backward()
        with torch.no_grad():

            # print("Intial: ", w, b)
            # print("Grad: ", w.grad)
            
            w = w - lr * w.grad

            # print("After: ", w, b)
            
            b = b - lr * b.grad


        w.requires_grad = True
        w.grad = None

        b.requires_grad = True
        b.grad = None

        # print("-"*120)

        if i % 10 == 0:
             print(f"Step {i:3d} | Loss: {loss.item():.6f}")

    return w.detach(), b.detach()

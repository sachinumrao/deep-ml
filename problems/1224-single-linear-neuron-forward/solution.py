import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # TODO: build nn.Linear(3, 1), set fixed weight/bias under no_grad, return float output

    layer1 = nn.Linear(3, 1)
    with torch.no_grad():
        layer1.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        layer1.bias.copy_(torch.tensor(0.1))

    out = layer1(x)
    return float(out.detach().numpy()[0][0])
    pass

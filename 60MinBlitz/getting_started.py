import torch
import numpy as np


def main():
    print("Create a 5x3 tensor with data from the memory space that it has been created from")
    x = torch.empty(5, 3)
    print(x)

    print("Create a 5x3 tensor with random data")
    x = torch.rand(5, 3)
    print(x)

    print("Create a zero filled matrix of data type long")
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    print("Create a tensor directly from data")
    x = torch.tensor([5.5, 3])
    print(x)

    print("Create a tensor based on an existing tensor")
    print("Create a tensor filled with ones")
    x = torch.ones(5, 3, dtype=torch.double)
    print(x)

    print("Create a tensor that has the same shape as before but filled with random data and adjusted data type")
    x = torch.randn_like(x, dtype=torch.float)
    print(x)

    print("Show the size")
    print(x.size())
    print()
    print("---------------------------------------------------------------------------------------------")
    print()

    print("OPERATIONS")
    print("Create a random tensor of the same size as before and add it to our previous tensor")
    y = torch.rand(5, 3)
    print(y)

    print()
    print(x + y)

    print("another way to add tensors")
    print(torch.add(x, y))

    print("providing an output tensor as argument")
    result = torch.empty(5, 3)
    print("non initialized empty tensor")
    print(result)
    torch.add(x, y, out=result)
    print("same tensor now used as the output for the addition operation")
    print(result)

    print("in place operation with .add_()")
    y.add_(x)
    print(y)

    print("using standard numpy indexing: x[:, 1] -> all rows and second column")
    print(x[:, 1])

    print("To resize, use .view and -1 will imply that torch will find automatically the other dimensions")
    x = torch.randn(4, 4)
    print(x)
    y = x.view(16)
    print(y)
    z = x.view(-1, 8)
    print("Printing the sizes of the tensors with the different resizes")
    print(x.size(), y.size(), z.size())

    print("For a one element tensor, use .item to get the item as a Python object")
    x = torch.randn(1)
    print(x)
    print(x.item())
    print()
    print("---------------------------------------------------------------------------------------------")
    print()

    print("numpy bridge")
    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    print("If we do operations on the tensor, it will change the value of the numpy array automatically")
    a.add_(1)
    print(a)
    print(b)

    print("we can create tensors from numpy arrays")
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

if __name__ == '__main__':
    main()

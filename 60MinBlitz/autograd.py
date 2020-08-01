import torch


def main():
    print("Create tensor where we track all operations on it")
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    print("Let's do tensor operation: y = x + 2")
    y = x + 2
    print(y)
    print(y.grad_fn)

    print("Do more operations on y: z = y * y * 3")
    z = y * y * 3
    out = z.mean()
    print(z, out)

    print("the .requires_grad_() method changes the requires_grad flag in place")
    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a * a).sum()
    print(b.grad_fn)

    print("Let's backprop...")
    out.backward()
    print(x.grad)

    print("Vector Jacobian product example")
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2

    print(y)

    v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(v)

    print(x.grad)

    print("Turning off autograd")
    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

    print("We can use .detach() to get a new tensor w/o autograd")
    print(x.requires_grad)
    y = x.detach()
    print(y.requires_grad)
    print(x.eq(y).all())


if __name__ == '__main__':
    main()

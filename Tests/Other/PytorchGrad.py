import torch
d = 5
x = torch.rand(d, requires_grad=True)
print('Tensor x:', x)
y = torch.ones(d, requires_grad=True)
print('Tensor y:', y)
loss = torch.sum(x*y)*3

del x
print()
print('Tracing back tensors:')
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

loss.backward()
getBack(loss.grad_fn)
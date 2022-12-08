import copy
from time import sleep

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib
import itertools
import numpy as np


pbar = tqdm(["a", "b", "c", "d"], desc='xjt')
for char in pbar:
    sleep(0.1)
    pbar.set_description("Processing %s" % char)

print('haha')

# 报错提醒
#try:
#    num=eval(input("请输入一个数："))
#    print(num**2)
#except NameError:
#    print("哈哈哈哈哈")

'''
测试深赋值
a = torch.tensor([[[1,2,3],[2,2,1],[2,32,4]],
                  [[1,2,3],[2,2,1],[1,2,4]]])
print(a.shape)
print(a,'a')
b = copy.deepcopy(a)
print(b,'b')
a[:,torch.arange(a.shape[1]), torch.arange(a.shape[2])] = 0
c = a
print(c,'c')
print(b,'b')
print(b.equal(c))
'''


'''
梯度下降法

x = torch.tensor([1., 2.], requires_grad=True)
# x: tensor([1., 2.], requires_grad=True)
y = 100*x
# y: tensor([100., 200.], grad_fn=<MulBackward0>)

loss = y.sum() # tensor(300., grad_fn=<SumBackward0>)

# Compute gradients of the parameters respect to the loss
print(x.grad)     # None, 反向传播前，梯度不存在
loss.backward()
print(x.grad)     # tensor([100., 100.]) loss对y的梯度为1， 对x的梯度为100

optim = torch.optim.SGD([x], lr=0.001) # 随机梯度下降， 学习率0.001
print(x)        # tensor([1., 2.], requires_grad=True)
optim.step()  # 更新x
print(x)        # tensor([0.9000, 1.9000], requires_grad=True) 变化量=梯度X学习率 0.1=100*0.001
'''


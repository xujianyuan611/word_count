import copy

import torch
import torch.nn as nn


class DistributionFitting(object):

    def __init__(self, model, optimizer, data_loader):
        """
        Creates a DistributionFitting object that summarizes all functionalities
        for performing the distribution fitting stage of ENCO.

        Parameters
        ----------
        model : MultivarMLP
                PyTorch module of the neural networks that model the conditional
                distributions.
        optimizer : torch.optim.Optimizer
                    Standard PyTorch optimizer for the model.
        data_loader : torch.data.DataLoader
                      Data loader returning batches of observational data. This
                      data is used for training the neural networks.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_module = nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
    # iter()的使用要结合后面的next()的使用，请看https://www.programiz.com/python-programming/methods/built-in/iter例子
        # self.hhh = 0

    def _get_next_batch(self):
        """
        Helper function for sampling batches one by one from the data loader.
        """
        # 我们的例子中，大部分走的是上面这条路，但也是有部分走的是下面这条路
        try:
            # self.hhh += 1
            batch = next(self.data_iter)
        except StopIteration:
            # print('你就看看是不是在40附近',self.hhh)
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        # 报错了就在重来一次，我估计是，就是，我已经验证过了。
        return batch

    def perform_update_step(self, sample_matrix):
        """
        Performs a full update step of the distribution fitting stage.
        It first samples a batch of random adjacency matrices from 'sample_matrix',
        and then performs a training step on a random observational data batch.

        Parameters
        ----------
        sample_matrix : torch.FloatTensor, shape [num_vars, num_vars]
                        Float tensor with values between 0 and 1. An element (i,j)
                        represents the probability of having an edge from X_i to X_j,
                        i.e., not masking input X_i for predicting X_j.

        Returns
        -------
        loss : float
               The loss of the model with the sampled adjacency matrices on the
               observational data batch.
        """
        batch = self._get_next_batch()
        # print(batch)
        # print('上面应该是本次用到采样数据？（我猜的），是的，我已经验证过了')
        # print(batch.shape)
        adj_matrices = self.sample_graphs(sample_matrix=sample_matrix,
                                          batch_size=batch.shape[0])
        # 这个的batch_size随便是多少都可以，只不过恰好有128这个数，已经够我们图形采样用了
        # print(type(adj_matrices))
        # print(adj_matrices.size(), '这个是在分布拟合中用到的邻接矩阵size，它是<class torch.Tensor')
        # print('我们看看它长什么样子：')
        # print(adj_matrices)  每一次分布拟合的迭代都不一样
        # print(batch)
        # print('哈哈哈哈哈哈哈哈哈哈')

        loss = self.train_step(batch, adj_matrices)

        return loss

    @torch.no_grad()
    def sample_graphs(self, sample_matrix, batch_size):
        """
        Samples a batch of adjacency matrices that are used for masking the inputs.
        """
        sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
        # print('111111111111')  就只是把邻接矩阵(概率版本)copy了 batch_size份
        # print(sample_matrix)
        # print('111111111111')
        adj_matrices = torch.bernoulli(sample_matrix)
        # print('111111111111')  # 就只是把邻接矩阵(概率版本)copy了 batch_size份
        # print(adj_matrices)
        # print('111111111111')
        # print('我想看看这个shape', adj_matrices.shape)
        # a = copy.deepcopy(adj_matrices)
        # Mask diagonals
        adj_matrices[:, torch.arange(adj_matrices.shape[1]), torch.arange(adj_matrices.shape[2])] = 0
        # b = adj_matrices
        # print(a == b, '咋不说话')
        # print(a.equal(b))
        # 经过验证 最后一行代码没有用，因为倒数第三行代码中，对角线的成功概率本来就是0，不可能因为倒数第二行施加了一下伯努利就会对角线出现1
        return adj_matrices

    def train_step(self, inputs, adj_matrices):
        """
        Performs single optimization step of the neural networks
        on given inputs and adjacency matrix.
        """
        # self.model 就是我们已经建立好的模型 请看 enco.py 的145行 我们要训练的就是它
        self.model.train()
        self.optimizer.zero_grad()  # 清除优化器中所有的梯度
        device = self.model.device
        # print(type(inputs), '看看这个类型是什么') 就是Tensor 这个to方法应该是pytorch 知道怎么用就行
        inputs = inputs.to(device)
        adj_matrices = adj_matrices.to(device)

        # Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrices = adj_matrices.transpose(1, 2)   # 为什么要把邻接矩阵的转置？我已经解决了，
        # 因为之前的邻接矩阵中，我们对于第(i,j)=1元素解读出来是j节点指向i节点，这么一个信息，把它转之后，第(i,j)=1元素解读出来是i节点指向j节点
        # 这个转置之后mask需要放到模型的forward里面进行计算，这和当时模型的计算概率的方法设定有关，更加清晰和明确。
        # print(type(inputs), '看看这个类型是什么') 还是Tensor
        preds = self.model(inputs, mask=mask_adj_matrices)
        # 我猜测，是由于前面的操作，比如self.model.train()，使得对self.model的直接输入，就会找到它的forward函数

        if inputs.dtype == torch.long:    # 走的是这条路
            # self.loss_module 就只是一个 nn.CrossEntropyLoss()
            # print(inputs.shape)       [128,8]    target 是每个样本的真实标签   这个的含义很明显  采集到的真实数据
            # print(preds.shape)         [128,8,10]  没有归一化的每个类的得分值(分越高越倾向分到这个类中)，它是通过在softmax(dim=1)后取log得到的得分值
            # print(preds.flatten(0, -2), '是不是预测值呢？')  并不是预测值 而是每个类的得分值     shape为 [1024,10]
            # print(inputs.reshape(-1),'是不是真实值呢？')   每个样本的真实取值      shape为 [1024]
            loss = self.loss_module(preds.flatten(0, -2), inputs.reshape(-1))
# 见文章 https://blog.csdn.net/claroja/article/details/108277017 和 https://zhuanlan.zhihu.com/p/383044774
            # 它是这样对应到我们原文中的损失函数去的：
            # 首先，取平均的分母是 128 * 8，对它进行解读含义没有什么意义，随便再乘什么数都可以，核心不在它这里。也不用在意之前我们熟悉的那种期望的求解方法
        # 之前的话总是 Eg(X) = \sum_x p(X=x)g(x) = 1/N \sum_{i=1}^N g(x_i)   QAQ 数理统计都忘到家了，本来就是用后面这个东西来估计前者的 相合估计
            # 其次，我们看这个计算步骤，对刚才经过神经网络计算得到的preds，softmax(dim=1)后，这个时候的含义是在本次神经网络的下，x随机变量的条件分布取值为各个类别值的概率
            # 再去根据索引，得到对应的值，就代表的是在本次神经网络的下，x随机变量的条件分布取值为真实值的概率，恰好对应文章中的 f_{\phi_i} (X_i;M_{-i} \odd X_{-i})
            # 然后，再去加log 加负号，恰好对应目标函数的核心部分。  实际上，我们的步骤是 先softmax(dim=1) 再log 再取出对应值 加负号

            # 最后前面的一些东西的含义也就迎刃而解了，比如，pred+softmax才是真正的完全的神经网络，从pred的求解过程中明白了神经网络的计算过程，再比如如何把输入父节点的参数放入到神经网络中

        else:  # If False, our input was continuous, and we return log likelihoods as preds
            loss = preds.mean()

        loss.backward()        # 损失函数backward，反向传播
        self.optimizer.step()  # 由更新参数组成的优化器step，更新参数（MLP各个层之间的权重、偏差值）

        return loss.item()

"""
This file contains the code for generating ground-truth conditional distributions.
Most experiments in the paper use the "NNCateg" distribution which is a randomly
initialized neural network.
"""
import numpy as np
import torch
import torch.nn as nn
from copy import copy
import sys
sys.path.append("../")
from causal_discovery.utils import get_device


class ProbDist(object):

    def __init__(self):
        """
        Abstract class representing a probability distribution. We want to sample from it, and
        eventually get the probability for an output.
        """
        pass

    def sample(self, inputs, batch_size=1):
        raise NotImplementedError
#raise NotImplementedError的意思是如果这个方法没有被子类重写，但是调用了，就会报错。


    def prob(self, inputs, output):
        raise NotImplementedError


####################
## DISCRETE PROBS ##
####################

class DiscreteProbDist(ProbDist):

    def __init__(self, val_range):
        """
        Abstract class of a discrete distribution (finite integer set or categorical).
        """
        super().__init__()
        self.val_range = val_range


class ConstantDist(DiscreteProbDist):

    def __init__(self, constant, val_range=None, **kwargs):
        """
        Represents a distribution that has a probability of 1.0 for one value, and zero for all others.
        """
        super().__init__(val_range=val_range)
        self.constant = constant

    def sample(self, inputs, batch_size=1):
        return np.repeat(self.constant, batch_size)

    def prob(self, inputs, output):
        if isinstance(output, np.ndarray):
            return (output == self.constant).astype(np.float32)
        else:
            return 1 if output == self.constant else 0

    def get_state_dict(self):
        # Export distribution
        state_dict = vars(self)
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = ConstantDist(state_dict["constant"], state_dict["val_range"])
        return obj


class CategoricalDist(DiscreteProbDist):

    def __init__(self, num_categs, prob_func, **kwargs):
        """
        Class representing a categorical distribution.

        Parameters
        ----------
        num_categs : int
                     Number of categories over which this distribution goes.
        prob_func :  这个玩意在代码中被调用的地方是在本页的倒数几行
                    object (LeafCategDist, CategProduct, IndependentCategProduct, or NNCateg)
                    Object that describes the mapping of input categories to output probabilities.
        """
        super().__init__(val_range=(0, num_categs))
        self.num_categs = num_categs
        self.prob_func = prob_func

    def sample(self, inputs, batch_size=1):
        # 为什么p的列维数是10呢？ 应该是这个随机变量选择在取值范围内取值为每个类别的概率
        print('这个节点的inputs', inputs, '它是这个节点的父节点的采集数据')
        p = self.prob_func(inputs, batch_size)

        # print('这个p是个啥呀',p)
        # print('看看p的维数',np.shape(p))
        # print('这个p[none]是个啥呀',p[None])  p[None]就是把那个一维的数组 外面再加一个数组符号

        if len(p.shape) == 1:
            p = np.repeat(p[None], batch_size, axis=0)
            # print('看看这个变化后的p',p)

        v = multinomial_batch(p)
        # print('看看这个v里面是什么',v)  采样选择概率最大的那个类别值
        return v

    def prob(self, inputs, output):
        p = self.prob_func(inputs, batch_size=1)
        while len(p.shape) > 2:
            p = p[0]
        if len(p.shape) == 2:
            return p[np.arange(output.shape[0]), output]
        else:
            return p[..., output]

    def get_state_dict(self):
        # Export distribution including prob_func details.
        state_dict = {"num_categs": self.num_categs}
        if self.prob_func is not None:
            state_dict["prob_func"] = self.prob_func.get_state_dict()
            state_dict["prob_func"]["class_name"] = str(self.prob_func.__class__.__name__)
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        if "prob_func" in state_dict:
            prob_func_class = None
            if state_dict["prob_func"]["class_name"] == "LeafCategDist":
                prob_func_class = LeafCategDist
            elif state_dict["prob_func"]["class_name"] == "CategProduct":
                prob_func_class = CategProduct
            elif state_dict["prob_func"]["class_name"] == "IndependentCategProduct":
                prob_func_class = IndependentCategProduct
            elif state_dict["prob_func"]["class_name"] == "NNCateg":
                prob_func_class = NNCateg
            prob_func = prob_func_class.load_from_state_dict(state_dict["prob_func"])
        else:
            prob_func = None
        obj = CategoricalDist(state_dict["num_categs"], prob_func)
        return obj


class LeafCategDist(object):

    def __init__(self, num_categs, scale=1.0):
        # num_categs 指的是这个随机变量的取值范围
        """
        Random categorical distribution to represent prior distribution of leaf nodes.
        """
        self.probs = _random_categ(size=(num_categs,), scale=scale)
        self.num_categs = num_categs

    def __call__(self, inputs, batch_size):
        return self.probs

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = LeafCategDist(state_dict["num_categs"])
        obj.probs = state_dict["probs"]
        return obj


class CategProduct(object):

    def __init__(self, input_names, input_num_categs=None, num_categs=None, val_grid=None, deterministic=False):
        """
        Categorical distribution with a random, independent distribution for every value pair of its parents.

        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        val_grid : np.ndarray, shape [input_num_categs[0], input_num_categs[1], ..., input_num_categs[-1], num_categs]
                   Array representing the probability distributions for each value pair of its parents. If 
                   None, a new val_grid is generated in this function.
        deterministic : bool
                        If True, we take the argmax over the generated val_grid, and assign a probability of
                        1.0 to the maximum value, all others zero.
        """
        if val_grid is None:
            assert input_num_categs is not None and num_categs is not None
            val_grid = _random_categ(size=tuple(input_num_categs) + (num_categs,))
            if deterministic:
                val_grid = (val_grid.max(axis=-1, keepdims=True) == val_grid).astype(np.float32)
        else:
            num_categs = val_grid.shape[-1]
            input_num_categs = val_grid.shape[:-1]
        self.val_grid = val_grid
        self.input_names = input_names
        self.input_num_categs = input_num_categs
        self.num_categs = num_categs

    def __call__(self, inputs, batch_size):
        idx = tuple([inputs[name] for name in self.input_names])
        v = self.val_grid[idx]
        return v

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = CategProduct(state_dict["input_names"],
                           state_dict["input_num_categs"],
                           state_dict["num_categs"])
        obj.val_grid = state_dict["val_grid"]
        return obj


class IndependentCategProduct(object):

    def __init__(self, input_names, input_num_categs, num_categs,
                 scale_prior=0.2, scale_val=1.0):
        """
        Represents the conditional distribution as a product of independent conditionals per parent.
        For instance, the distribution p(A|B,C) is represented as:
                    p(A|B,C)=p_A(A)*p_B(A|B)*p_C(A|C)/sum_A[p_A(A)*p_B(A|B)*p_C(A|C)]
        
        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        scale_prior : float
                      Scale of the _random_categ distribution to use for the prior p_A(A)
        scale_val : float
                    Scale of the _random_categ distribution to use for all conditionals.
        """
        num_inputs = len(input_names)
        val_grids = [_random_categ(size=(c, num_categs), scale=scale_val) for c in input_num_categs]
        prior = _random_categ((num_inputs,), scale=scale_prior)
        self.val_grids = val_grids
        self.prior = prior
        self.num_categs = num_categs
        self.input_names = input_names
        self.input_num_categs = input_num_categs

    def __call__(self, inputs, batch_size):
        probs = np.zeros((batch_size, self.num_categs))
        for idx, name in enumerate(self.input_names):
            probs += self.prior[idx] * self.val_grids[idx][inputs[name]]
        return probs

    def get_state_dict(self):
        state_dict = copy(vars(self))
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = IndependentCategProduct(state_dict["input_names"],
                                      state_dict["input_num_categs"],
                                      state_dict["num_categs"])
        obj.prior = state_dict["prior"]
        obj.val_grids = state_dict["val_grids"]
        return obj


class NNCateg(object):

    def __init__(self, input_names, input_num_categs, num_categs):
        """
        Randomly initialized neural network that models an arbitrary conditional distribution.
        The network consists of a 2-layer network with LeakyReLU activation and an embedding
        layer for representing the categorical inputs. Weights are initialized with the 
        orthogonal initialization, and the biases uniform between -0.5 and 0.5.
        Architecture and initialization widely taken from Ke et al. (2020).

        Parameters
        ----------
        input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
        input_num_categs : list[int]
                           Number of categories each input variable can take.
        num_categs : int
                     Number of categories over which the conditional distribution should be. 
        """
        num_hidden = 48
        embed_dim = 4
        # print('看一看input_num_categs是什么：',input_num_categs,sum(input_num_categs))
        # nn.Embedding输入的第一个元素是 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999，在这里我们有时候是10or20
        self.embed_module = nn.Embedding(sum(input_num_categs), embed_dim)
        # print(self.embed_module.weight)
        # 应该是拿向量的4维分量代表一个节点  有几个父节点，就有几个父节点*4个输入特征值

        # print('这是在哪一步进行的操作') 原来一切都比我想象的早， 在第一步，建立图的时候就出现了！！！
        self.net = nn.Sequential(nn.Linear(embed_dim*len(input_num_categs), num_hidden),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(num_hidden, num_categs, bias=False),
                                 nn.Softmax(dim=-1))
        # 第二行，0.1这个东西是输入值小于0时候才会去使用的"负斜率"

        # 第三行，当bias为false的时候，就不加常数项
# https://www.bilibili.com/video/BV1hE411t7RN/?p=21&spm_id_from=pageDriver&vd_source=46e39f15fcea0c639c6361c9ba18d485

        # x的维数是输入的特征维数 * 1的向量，A的维数是输入的维数 * 输入的特征维数
        # 框架是现有一个全连接的线性神经网络， A * x.T + b ，现在维数从输入的特征维数embed_dim*len(input_num_categs)变成了隐藏层的维数
        # 然后对每个值经过一个非线性激活函数LeakyReLU，但是维数不变，再有一个全连接的线性神经网络（无偏差），A * x.T ，现在维数从隐藏层的维数变成该随机变量取值种类数的维数
        # 最后就是进行一下softmax，维数不变，现在的结果就代表该随机变量输出各个值的概率

        for name, p in self.net.named_parameters():  # 在我们的例子中，进行了三次循环，分别是第1层的权重矩阵，第1层的偏差向量，第3层的权重矩阵
            # print(name, p.size())                                           # p 就代表上面的A矩阵 or 偏差向量
            # print(p)              没有训练的p是服从官方文档给出的均匀分布[-1/b,1/b] b=输入层的维数  也就是说对于只有一个父节点的bais层不用重新赋值也可以
            # print(p,'1111111111111')
            if name.endswith(".bias"):    # 就只是对于第1层的bias部分走的是这条路
                # print('mmmmmmmmmmmmmm')
                p.data.uniform_(-0.5, 0.5)
                # print(p, '222222222')
            else:                         # 对第1层的权重矩阵部分，和第3层的权重矩阵部分走的是这条路
                nn.init.orthogonal_(p, gain=2.5)
                # 把p这个size的tensor里面的元素重新赋值给p
                # gain是什么意思？ 是这样子的，如果gain=1(默认)，那么，利用 a = nn.init.orthogonal_(p, gain=1) 得到的a
                # torch.mm(a,a.t())就是单位矩阵，a就是半正交矩阵      ，如果gain = 2.5,那么torch.mm(a,a.t())就是6.25 * 单位矩阵
                # 如果gain = 2,那么torch.mm(a,a.t())就是4 * 单位矩阵 ，如果gain = 3,那么torch.mm(a,a.t())就是9 * 单位矩阵
        # print('robinnnnnnnnnnn!')
        # print(self.net)
        # for i in self.net.parameters():
        #    print(i.size())
            # 我们发现，上面两种方式的赋值，是定义节点和父节点之间的真实的条件概率，所以都是很随意的，任由我们想怎么赋值就怎么赋值。
            # 他作为我们产生数据的真实分布，将用于产生条件分布下的采样，为什么说"将"呢？因为还没有施加父节点数据
            # 关键是看后面训练神经网络的时候，会把重新给定的初值的权重，偏差一个个的都给修正的好好的，使得能够很好的拟合上面我们设置的条件分布。
        self.num_categs = num_categs
        self.input_names = input_names
        self.input_num_categs = input_num_categs
        self.device = get_device()
        self.embed_module.to(self.device)
        self.net.to(self.device)

    @torch.no_grad()
    def __call__(self, inputs, batch_size):
        inp_tensor = None
        for i, n, categs in zip(range(len(self.input_names)), self.input_names, self.input_num_categs):
            v = torch.from_numpy(inputs[n]).long()+sum(self.input_num_categs[:i])
            v = v.unsqueeze(dim=-1)
            inp_tensor = v if inp_tensor is None else torch.cat([inp_tensor, v], dim=-1)
        inp_tensor = inp_tensor.to(self.device)
        inp_tensor = self.embed_module(inp_tensor).flatten(-2, -1)    # 利用了embedding层的输入
        probs = self.net(inp_tensor).cpu().numpy()
        return probs
        # 果然没有错，debug在这里我就能得到结论了，注意，第一个节点很随机的采样的数据是[8,9,8,9,9,5,1,4,8,8,8]，然后我debug到这里后发现果然第二个节点在各个分类值的取值概率中，第2、4和5行概率值完全一样，第1和3行概率值完全一样
        # 这里的操作就实现了根据因果图来产生数据，借用的是我们在上面(就这个类中的init那块)，

    def get_state_dict(self):
        state_dict = copy(vars(self))
        state_dict["embed_module"] = self.embed_module.state_dict()
        state_dict["net"] = self.net.state_dict()
        return state_dict

    @staticmethod
    def load_from_state_dict(state_dict):
        obj = NNCateg(state_dict["input_names"],
                      state_dict["input_num_categs"],
                      state_dict["num_categs"])
        obj.embed_module.load_state_dict(state_dict["embed_module"])
        obj.net.load_state_dict(state_dict["net"])
        return obj

def multinomial_batch(p):
    # Effient batch-scale sampling in numpy
    u = np.random.uniform(size=p.shape[:-1]+(1,))
    p_cumsum = np.cumsum(p, axis=-1)
    # 上面这个函数np.cumsum的作用就是，把p的每一行，逐列相加。在我们的例子中，对于根节点的p, p =[[0.02,0.20,0.04,0.05,0.09,0.07,0.02,0.04,0.24,0.19],后面全和第一个一样] ,p_cumsum = [[0.02,0.22,0.26,0.32,0.41,0.49,0.51,0.55,0.80,1.0],后面全和第一个一样]
    diff = (p_cumsum - u)
    diff[diff < 0] = 2  # Set negatives to any number larger than 1
    samples = np.argmin(diff, axis=-1)
    print('这个节点的采样值为：', samples)
    return samples
# 我发现，对于根节点，产生数据的随机性，就是，很随机的产生采样数据而已。
# 我发现，对于除了根节点的所有节点，它是这样子生成采样数据的。根据父节点的采样值是多少，这个节点产生各个类比的概率就是多少，如果只是按照这样子就采样的话，相当于是一一对应了，父节点是a，那么这个节点的采样值一定是b
# 这样子不够合理，我们只需要对产生各个类比的概率值添加一些扰动，展现出采样的随机性就能够做到，模拟现实生活中的采样。

######################
## CONTINUOUS PROBS ##
######################

class ContinuousProbDist(ProbDist):

    def __init__(self):
        """
        Template class for continuous probability distributions.
        """
        super().__init__()


#####################
## DIST GENERATORS ##
#####################

def _random_categ(size, scale=1.0, axis=-1):
    """
    Returns a random categorical distribution by sampling a value from a Gaussian distribution for each category, 
    and applying a softmax on those.

    Parameters
    ----------
    size - int / tuple
           For integer: Number of categories over which the distribution should be.
           For tuple: array size of samples from the Gaussian distribution
    scale - float
            Standard deviation to use for the Gaussian to sample from. scale=0.0 corresponds to a uniform 
            distribution. The larger the scale, the more peaked the distribution will be.
    axis - int
           If size is a tuple, axis specifies which axis represents the categories. The softmax is applied
           over the axis dimension.
    """
    # print('我想看看size是元组还是整数', size)  # 但是却是(10,) 这个元组
    val_grid = np.random.normal(scale=scale, size=size)  # 默认就是均值为0 标准差为1的正态分布
    # print('应该是十个正态随机数',val_grid)
    # 看了一下关于np.random.normal的函数说明，这个size就是代表，我们要输出产生几维数的正态随机变量

    val_grid = np.exp(val_grid)   # 广播机制

    val_grid = val_grid / val_grid.sum(axis=axis, keepdims=True)
    # print('应该是这个随机变量取各个值的概率', val_grid)

    # 可是为什么要这么去模仿，也只能是模型的假设？ 这个随机变量产生怎样的数据，应该是由数据本身来决定的。
    # 但这么做也没有关系，等到实际把该ENCO算法应用到现实数据中，我们就需要先估计一下根节点产生各个随机变量的概率，替换这里的概率
    return val_grid


def get_random_categorical(input_names, input_num_categs, num_categs, inputs_independent=True, use_nn=False, deterministic=False, **kwargs):
    """
    Returns a randomly generated, conditional distribution for categorical variables.

    Parameters
    ----------
    input_names : list[str]
                  List of variable names that are supposed to be the parents in this conditional distribution.
                  Use an empty list to denote a leaf node distribution.
    input_num_categs : list[int]
                       Number of categories each input variable can take.
    num_categs : int
                 Number of categories over which the conditional distribution should be. 
    inputs_independent : bool
                         If True and not use_nn and not deterministic, the distribution is an IndependentCategProduct,
                         which models the distribution as product of independent conditionals.
    use_nn : bool
             If True and not deterministic, a randomly initialized neural network is used for generating the distribution.
    deterministic : bool
                    If True, the returned deterministic distribution will be deterministic. This means for every value
                    pair of the conditionals, there exists one category which has a probability 1.0, and all other
                    categories have a zero probability.
    """
    num_inputs = len(input_names)

    if num_inputs == 0:   # 在我们的例子中，当父节点是空集时候，这个随机变量的条件分布的定义走的是这条路，感觉这个命名有点问题，应该是根节点而不是叶节点
        prob_func = LeafCategDist(num_categs)
        # print('这个根节点的条件分布等价于它的数据分布(由我们怎么想弄怎么弄)', prob_func.probs)   他将用于我们产生数据的真实分布，但并不是现在这个权重，因为还没有施加父节点数据
        # print('xxxxxxxxxxxxxxxxxxxx5')                                                但是由于这个是根节点，所以没有关系，应该用的还是这个数据
        print('父节点的name：', input_names, '父节点的input_num_categs取值范围', input_num_categs, '这个随机变量的取值范围', num_categs)


    elif deterministic:
        prob_func = CategProduct(input_names, input_num_categs, num_categs, deterministic=deterministic)
        # print('xxxxxxxxxxxxxxxxxxxx4')

    elif use_nn:    # 在我们的例子中，当父节点是非空集时候，这个随机变量的条件分布的定义走的是这条路

        prob_func = NNCateg(input_names, input_num_categs, num_categs)

        print('父节点的name：', input_names, '父节点的input_num_categs取值范围', input_num_categs, '这个随机变量的取值范围', num_categs)

        # 现在我还是很好奇一个问题，每一行的数据真的是同一次根据因果图产生的产生的吗，因为目前根节点的分布好说，其他随机变量的条件分布我正在看怎么弄
        # 现在这里还并无法知道这个答案，现在只是定义好了每个随机变量的父节点的不同个数，来构建出的神经网络，具体训练我们后面才看
        # 或许他将用于我们产生数据的真实分布，但并不是现在这个权重，因为还没有施加父节点数据

        # print('为什么不说话')
        # 我终于找到你这个 prob_func 东西了 用来对条件概率进行分布拟合
        # 这也是一个类，不是一个函数
        # print('xxxxxxxxxxxxxxxxxxxx3')

    elif inputs_independent:
        prob_func = IndependentCategProduct(input_names, input_num_categs, num_categs)
        # print('xxxxxxxxxxxxxxxxxxxx2')

    else:
        prob_func = CategProduct(input_names, input_num_categs, num_categs)
        # print('xxxxxxxxxxxxxxxxxxxx1')

    # print('xxxxxxxxxxxxxxxxxxxx')
    return CategoricalDist(num_categs, prob_func, **kwargs)





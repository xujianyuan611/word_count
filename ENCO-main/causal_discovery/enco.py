import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import sys
sys.path.append("../")

from causal_discovery.distribution_fitting import DistributionFitting
from causal_discovery.utils import track, find_best_acyclic_graph
from causal_discovery.multivariable_mlp import create_model
from causal_discovery.multivariable_flow import create_continuous_model
from causal_discovery.graph_fitting import GraphFitting
from causal_discovery.datasets import ObservationalCategoricalData
from causal_discovery.optimizers import AdamTheta, AdamGamma



class ENCO(object):

    def __init__(self, 
                 graph,
                 hidden_dims=[64],
                 use_flow_model=False,
                 lr_model=5e-3,
                 betas_model=(0.9, 0.999),
                 weight_decay=0.0,
                 lr_gamma=2e-2,
                 betas_gamma=(0.9, 0.9),
                 lr_theta=1e-1,
                 betas_theta=(0.9, 0.999),
                 model_iters=1000,
                 graph_iters=100,
                 batch_size=128,
                 GF_num_batches=1,
                 GF_num_graphs=100,
                 lambda_sparse=0.004,
                 latent_threshold=0.35,
                 use_theta_only_stage=False,
                 theta_only_num_graphs=4,
                 theta_only_iters=1000,
                 max_graph_stacking=200,
                 sample_size_obs=5000,
                 sample_size_inters=200):
        """
        Creates ENCO object for performing causal structure learning.

        Parameters
        ----------
        graph : CausalDAG
                The causal graph on which we want to perform causal structure learning.
        hidden_dims : list[int]   # [第1个隐藏层的维数,...,第n个隐藏层的维数]
                      Hidden dimensionalities to use in the distribution fitting neural networks.
                      Listing more than one dimensionality creates multiple hidden layers.
        lr_model : float
                   Learning rate to use in distribution fitting stage for the neural networks.
        betas_model : tuple (float, float)
                      Beta values to use for the model's Adam optimizer.
        weight_decay : float
                       Weight decay to use in the model optimizer during the graph fitting stage.
        lr_gamma : float
                   Learning rate to use in the graph fitting stage for the gamma parameters.
        betas_gamma : tuple (float, float)
                      Beta values to use for the gamma's Adam optimizer.
        lr_theta : float
                   Learning rate to use in the graph fitting stage for the theta parameters.
        betas_theta : tuple (float, float)
                      Beta values to use for the theta's Adam optimizer.
        model_iters : int              # 在分布拟合中，模型迭代的次数
                      Number of update steps to perform in each distribution fitting stage.
        graph_iters : int              # 在图形拟合中，模型迭代的次数
                      Number of update steps to perform in each graph fitting stage.
        batch_size : int
                     Batch size to use in both distribution and graph fitting stage.
        GF_num_batches : int
                         Number of batches to use per MC sample in the graph fitting stage.
                         Usually 1, only higher needed if GPU is running out of memory for 
                         common batch sizes.
        GF_num_graphs : int 
                        Number of graph samples to use for estimating the gradients in the 
                        graph fitting stage. Usually in the range 20-100.
        lambda_sparse : float
                        Sparsity regularizer value to use in the graph fitting stage.
        latent_threshold : float
                           Threshold to use for latent confounder detection. Correspond to the
                           hyperparameter tau in the paper.
        use_theta_only_stage : bool
                               If True, gamma is frozen in every second graph fitting step, and
                               more sample-efficient gradient estimators can be used for theta.
                               Is only necessary for large graphs. Details about this stage
                               are described in Appendix D.2
        theta_only_num_graphs : int
                                Number of graph samples to use in the graph fitting stage if gamma
                                is frozen. Needs to be an even number, and usually 2 or 4.
        theta_only_iters : int
                           Number of update steps to perform in each graph fitting stage if
                           gamma is frozen. Can be much higher than graph_iters since less
                           graph samples are needed per update step. 
        max_graph_stacking : int
                             Number of graphs that can maximally evaluated in parallel on the device
                             during the graph fitting stage. If you run out of GPU memory, try to 
                             lower this number. The graphs will then be evaluated in sequence, which 
                             can be slightly slower but uses less memory.
        sample_size_obs: int
                         Dataset size to use for observational data. If an exported graph is
                         given as input and sample_size_obs is smaller than the exported
                         observational dataset, the first sample_size_obs samples will be taken.
        sample_size_inters: Number of samples to use per intervention. If an exported graph is
                            given as input and sample_size_inters is smaller than the exported
                            interventional dataset, the first sample_size_inters samples will be taken.
        exclude_inters : list
                         A list of variable indices that should be excluded from sampling interventions
                         from. This should be used to apply ENCO on intervention sets on a subset of 
                         the variable set. If None, an empty list will be assumed, i.e., interventions
                         on all variables will be used.
        """
        self.graph = graph
        self.num_vars = graph.num_vars    # 这个传入的graph参数还得是一个CausalDAG类呢 graph.num_vars代表因果图上随机变量的个数

# nn.Sequential(nn.Linear(embed_dim * len(input_num_categs), num_hidden),
#              nn.LeakyReLU(0.1),
#              nn.Linear(num_hidden, num_categs, bias=False),
#              nn.Softmax(dim=-1))
        #   上面这个就是产生真实数据的模型        其中， num_hidden = 48  embed_dim = 4  input_num_categs = 10

        # Create observational dataset

        obs_dataset = ObservationalCategoricalData(graph, dataset_size=sample_size_obs)
        # obs_dataset.data就是采样的真实数据
        # print(obs_dataset.data.shape, '真实数据，应该是五千条')
        # print('1111111应该是观测数据采样结束了')

        # 然后，为了在神经网络拟合中用到这个数据，就需要转化到下面这个obs_data_loader，它还就是之前的obs_dataset，
        # 只不过将obs_dataset能够分成一组一组，来为之后训练模型做准备， 例如，obs_dataset的shape是(5000,8)，之后的一组组shape是(128,8)
        obs_data_loader = data.DataLoader(obs_dataset, batch_size=batch_size,
                                          shuffle=True, drop_last=True)

        # Create neural networks for fitting the conditional distributions
        if graph.is_categorical:
            num_categs = max([v.prob_dist.num_categs for v in graph.variables])
            # num_categs就是所有分类随机变量的取值种数最多的那个

            model = create_model(num_vars=self.num_vars,
                                 num_categs=num_categs,
                                 hidden_dims=hidden_dims)  # 这个的隐藏层可以和产生数据的隐藏层不一样

        else:
            model = create_continuous_model(num_vars=self.num_vars,
                                            hidden_dims=hidden_dims,
                                            use_flow_model=use_flow_model)

        # 让我们先把在模型建立部分干的事情弄明白再去进行分布拟合吧，也就是建立了下面这个模型，和产生数据的模型不同的是，我们要对下面这个模型进行训练
# MultivarMLP(
#   (layers): ModuleList(
#     (0): EmbedLayer(
#       (input_mask): InputMask()
#       (embedding): Embedding(640, 64)
#     )
#     (1): LeakyReLU(negative_slope=0.1, inplace=True)
#     (2): MultivarLinear(input_dims=64, output_dims=64, extra_dims=[8])
#     (3): LeakyReLU(negative_slope=0.1, inplace=True)
#     (4): MultivarLinear(input_dims=64, output_dims=10, extra_dims=[8])
#   )
# )

        # 要进行训练的准备步骤
        model_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=lr_model,
                                           betas=betas_model,
                                           weight_decay=weight_decay)
        # Initialize graph parameters
        self.init_graph_params(self.num_vars, lr_gamma, betas_gamma, lr_theta, betas_theta)


        # Initialize distribution and graph fitting modules
        self.distribution_fitting_module = DistributionFitting(model=model,
                                                               optimizer=model_optimizer,
                                                               data_loader=obs_data_loader)

        # 在干预数据中采样的过程，是在graph fitting进行的，确实也是，这个时候才需要进行
        self.graph_fitting_module = GraphFitting(model=model,
                                                 graph=graph,
                                                 num_batches=GF_num_batches,
                                                 num_graphs=GF_num_graphs,
                                                 theta_only_num_graphs=theta_only_num_graphs,
                                                 batch_size=batch_size,
                                                 lambda_sparse=lambda_sparse,
                                                 max_graph_stacking=max_graph_stacking,
                                                 sample_size_inters=sample_size_inters,
                                                 exclude_inters=self.graph.exclude_inters)

        # Save other hyperparameters
        self.model_iters = model_iters
        self.graph_iters = graph_iters
        self.use_theta_only_stage = use_theta_only_stage
        self.theta_only_iters = theta_only_iters
        self.latent_threshold = latent_threshold
        self.true_adj_matrix = torch.from_numpy(graph.adj_matrix).bool()
        self.true_node_relations = torch.from_numpy(graph.node_relations)
        self.metric_log = []
        self.iter_time = -1
        self.dist_fit_time = -1

        # Some debugging info for user
        print(f'Distribution fitting model:\n{str(model)}')
        print(f'Dataset size:\n- Observational: {len(obs_dataset)}\n- Interventional: {sample_size_inters}')

    def init_graph_params(self, num_vars, lr_gamma, betas_gamma, lr_theta, betas_theta):
        """
        Initializes gamma and theta parameters, including their optimizers.
        """
        self.gamma = nn.Parameter(torch.zeros(num_vars, num_vars))  # Init with zero => prob 0.5
        # nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter，并且会向宿主模型注册该参数，成为一部分
        # graph.gamma 指的就是\gamma矩阵，边存在矩阵
        self.gamma.data[torch.arange(num_vars), torch.arange(num_vars)] = -9e15  # Mask diagonal
        # 因为模型的假设是不包含自循环，对角线上参数的不进行梯度更新，恒为0

        # For latent confounders, we need to track interventional and observational gradients separat => different opt
        if self.graph.num_latents > 0:
            self.gamma_optimizer = AdamGamma(self.gamma, lr=lr_gamma, beta1=betas_gamma[0], beta2=betas_gamma[1])
        else:
            self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=lr_gamma, betas=betas_gamma)

        self.theta = nn.Parameter(torch.zeros(num_vars, num_vars))  # Init with zero => prob 0.5
        self.theta_optimizer = AdamTheta(self.theta, lr=lr_theta, beta1=betas_theta[0], beta2=betas_theta[1])

    def discover_graph(self, num_epochs=30, stop_early=False):
        """
        Main training function. It starts the loop of distribution and graph fitting.
        Returns the predicted binary adjacency matrix.
        """
        num_stops = 0
        # num_epochs 是算法的迭代次数；model_iters是在分布拟合中，模型迭代的次数；graph_iters是在图形拟合中，模型迭代的次数
        for epoch in track(range(num_epochs), leave=False, desc="Epoch loop"):
            self.epoch = epoch
            start_time = time.time()
            # Update Model
            self.distribution_fitting_step()    # 进行"distribution fitting"
            self.dist_fit_time = time.time() - start_time
            # Update graph parameters
            self.graph_fitting_step()           # 进行"graph fitting"
            self.iter_time = time.time() - start_time
            # Print stats
            self.print_graph_statistics(epoch=epoch+1, log_metrics=True)
            # Early stopping if perfect reconstruction for 5 epochs (for faster prototyping)
            if stop_early and self.is_prediction_correct():
                num_stops += 1
                if num_stops >= 5:
                    print("Stopping early due to perfect discovery")
                    break
            else:
                num_stops = 0
        return self.get_binary_adjmatrix()

    def distribution_fitting_step(self):
        """
        Performs on iteration of distribution fitting.
        """
        # Probabilities to sample input masks from
        sample_matrix = torch.sigmoid(self.gamma) * torch.sigmoid(self.theta)
        #  采用上次graph fitting迭代后的的gamma和theta矩阵的值来生成 邻接矩阵(概率版本)
        # 例如，下面的左边矩阵是第一次用gamma和theta矩阵的值来生成邻接概率矩阵，右边矩阵是第二次用gamma和theta矩阵的值(第一次graph fitting后迭代值)来生成邻接概率矩阵
        '''
                                            tensor([[0.0000, 0.7944, 0.7923, 0.3737, 0.3459, 0.1343, 0.1559, 0.7953],
                                                    [0.0579, 0.0000, 0.2167, 0.7886, 0.7806, 0.4121, 0.2359, 0.2477],
             左边这个矩阵,                            [0.0648, 0.2273, 0.0000, 0.3274, 0.7621, 0.1988, 0.1698, 0.2513],
         也就是右边这个矩阵 除了对角线上元素全是0.25 ,     [0.0805, 0.0656, 0.1505, 0.0000, 0.2229, 0.7847, 0.7781, 0.1423],
                                                    [0.0945, 0.0573, 0.0578, 0.1704, 0.0000, 0.2146, 0.7771, 0.0990],
                                                    [0.2451, 0.1262, 0.1315, 0.0692, 0.1759, 0.0000, 0.2479, 0.0478],
                                                    [0.2055, 0.1819, 0.1963, 0.0663, 0.0711, 0.1223, 0.0000, 0.7830],
                                                    [0.0581, 0.1642, 0.1261, 0.2841, 0.2848, 0.3734, 0.0646, 0.0000]],
                                           grad_fn=<MulBackward0>)
        '''

        # print(sample_matrix)
        # Update model in a loop
        t = track(range(self.model_iters), leave=False, desc="Distribution fitting loop")
        # print(t, 't原来是这个东西')
        for _ in t:    # 这个loss步骤中，就是通过不断调整神经网络，去拟合之前我们设定的真实条件分布
            loss = self.distribution_fitting_module.perform_update_step(sample_matrix=sample_matrix)  # 邻接矩阵(概率版本）
            # print('问题： 研究这个loss 他到底把损失函数扔到哪里去了 QAQ，现在我找到了，就差说明这个和原文中的交叉熵是一个东西了') 搞定 请看distribution_fitting中138行
            if hasattr(t, "set_description"):
                t.set_description("Model update loop, loss: %4.2f" % loss)    # %字段宽度.保留小数点后几位，是字符串格式化。而且你看它是修改了t的desc部分

    def graph_fitting_step(self):
        """
        Performs on iteration of graph fitting.
        """
        # For large graphs, freeze gamma in every second graph fitting stage
        only_theta = (self.use_theta_only_stage and self.epoch % 2 == 0)
        iters = self.graph_iters if not only_theta else self.theta_only_iters
        # Update gamma and theta in a loop
        for _ in track(range(iters), leave=False, desc="Graph fitting loop"):
            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            theta_mask, var_idx = self.graph_fitting_module.perform_update_step(self.gamma,
                                                                                self.theta,
                                                                                only_theta=only_theta)
            if not only_theta:  # In the gamma freezing stages, we do not update gamma
                if isinstance(self.gamma_optimizer, AdamGamma):
                    self.gamma_optimizer.step(var_idx)
                else:
                    self.gamma_optimizer.step()
            self.theta_optimizer.step(theta_mask)

    def get_binary_adjmatrix(self):
        """
        Returns the predicted, binary adjacency matrix of the causal graph.
        """
        binary_gamma = self.gamma > 0.0
        # 相当于是它们的sigmiod函数大于0.5了
        binary_theta = self.theta > 0.0
        A = binary_gamma * binary_theta
        # If we consider latent confounders, we mask all edges that have a confounder score greater than the threshold
        if self.graph.num_latents > 0:
            A = A * (self.get_confounder_scores() < self.latent_threshold)
            
        return (A == 1).cpu()

    def get_acyclic_adjmatrix(self):
        """
        Returns the predicted, acyclic adjacency matrix of the causal graph.
        """
        A = find_best_acyclic_graph(gamma=torch.sigmoid(self.gamma), 
                                    theta=torch.sigmoid(self.theta))
        return A.cpu()

    def is_prediction_correct(self):
        """
        Returns true if the prediction corresponds to the correct, underlying causal graph. Otherwise false.
        If latent confounders exist, those need to be correct as well to return true.
        """
        correct_pred = (self.get_binary_adjmatrix() == self.true_adj_matrix).all()
        if self.graph.num_latents > 0:
            conf_metrics = self.get_confounder_metrics()
            correct_pred = correct_pred and ((conf_metrics["FP"]+conf_metrics["FN"]) == 0)
        return correct_pred

    def get_confounder_scores(self):
        """
        Returns a matrix of shape [num_vars, num_vars] where the element (i,j) represents the confounder score
        between the variable pair X_i and X_j, i.e., lc(X_i,X_j).
        """
        if isinstance(self.gamma_optimizer, AdamGamma):
            gamma_obs_sig, gamma_int_sig = torch.unbind(torch.sigmoid(self.gamma_optimizer.updates), dim=-1)
            l_score = gamma_obs_sig * (1 - gamma_int_sig)
            l_score *= l_score.T  # Scores are a symmetric matrix
        else:
            l_score = torch.zeros_like(self.gamma)
        return l_score

    @torch.no_grad()
    def print_graph_statistics(self, epoch=-1, log_metrics=False, m=None):
        """
        Prints statistics and metrics of the current graph prediction. It is executed
        during training to track the training progress.
        """
        if m is None:
            m = self.get_metrics()
        if log_metrics:
            if epoch > 0:
                m["epoch"] = epoch
            self.metric_log.append(m)

        if epoch > 0:
            print("--- [EPOCH %i] ---" % epoch)
        print("Graph - SHD: %i, Recall: %4.2f%%, Precision: %4.2f%% (TP=%i,FP=%i,FN=%i,TN=%i)" %
              (m["SHD"], 100.0*m["recall"], 100.0*m["precision"], m["TP"], m["FP"], m["FN"], m["TN"]))
        print("      -> FP:", ", ".join(["%s=%i" % (key, m["FP_details"][key]) for key in m["FP_details"]]))
        print("Theta - Orientation accuracy: %4.2f%% (TP=%i,FN=%i)" %
              (m["orient"]["acc"] * 100.0, m["orient"]["TP"], m["orient"]["FN"]))

        if self.graph.num_latents > 0 and "confounders" in m:
            print("Latent confounders - TP=%i,FP=%i,FN=%i,TN=%i" %
                  (m["confounders"]["TP"], m["confounders"]["FP"], m["confounders"]["FN"], m["confounders"]["TN"]))

        if epoch > 0 and self.num_vars >= 100:  # For large graphs, we print runtime statistics for better time estimates
            print("-> Iteration time: %imin %is" % (int(self.iter_time)//60, int(self.iter_time) % 60))
            print("-> Fitting time: %imin %is" % (int(self.dist_fit_time)//60, int(self.dist_fit_time) % 60))
            gpu_mem = torch.cuda.max_memory_allocated(device="cuda:0")/1.0e9 if torch.cuda.is_available() else -1
            print("-> Used GPU memory: %4.2fGB" % (gpu_mem))

    @torch.no_grad()
    def get_metrics(self, enforce_acyclic_graph=False):
        """
        Returns a dictionary with detailed metrics comparing the current prediction to the ground truth graph.
        """
        # Standard metrics (TP,TN,FP,FN) for edge prediction
        binary_matrix = self.get_binary_adjmatrix()
        if enforce_acyclic_graph:
            binary_matrix = self.get_acyclic_adjmatrix()
        else:
            binary_matrix = self.get_binary_adjmatrix()
        false_positives = torch.logical_and(binary_matrix, ~self.true_adj_matrix)
        false_negatives = torch.logical_and(~binary_matrix, self.true_adj_matrix)
        TP = torch.logical_and(binary_matrix, self.true_adj_matrix).float().sum().item()
        TN = torch.logical_and(~binary_matrix, ~self.true_adj_matrix).float().sum().item()
        FP = false_positives.float().sum().item()
        FN = false_negatives.float().sum().item()
        TN = TN - self.gamma.shape[-1]  # Remove diagonal as those are not being predicted
        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)
        # Structural Hamming Distance score
        rev = torch.logical_and(binary_matrix, self.true_adj_matrix.T)
        num_revs = rev.float().sum().item()
        SHD = (false_positives + false_negatives + rev + rev.T).float().sum().item() - num_revs

        # Get details on False Positives (what relations have the nodes of the false positives?)
        FP_elems = torch.where(torch.logical_and(binary_matrix, ~self.true_adj_matrix))
        FP_relations = self.true_node_relations[FP_elems]
        FP_dict = {
            "ancestors": (FP_relations == -1).sum().item(),  # i->j => j is a child of i
            "descendants": (FP_relations == 1).sum().item(),
            "confounders": (FP_relations == 2).sum().item(),
            "independents": (FP_relations == 0).sum().item()
        }

        # Details on orientation prediction of theta, independent of gamma
        orient_TP = torch.logical_and(self.true_adj_matrix == 1, self.theta.cpu() > 0.0).float().sum().item()
        orient_FN = torch.logical_and(self.true_adj_matrix == 1, self.theta.cpu() <= 0.0).float().sum().item()
        orient_acc = orient_TP / max(1e-5, orient_TP + orient_FN)
        orient_dict = {
            "TP": int(orient_TP),
            "FN": int(orient_FN),
            "acc": orient_acc
        }

        # Summarizing all results in single dictionary
        metrics = {
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "SHD": int(SHD),
            "reverse": int(num_revs),
            "recall": recall,
            "precision": precision,
            "FP_details": FP_dict,
            "orient": orient_dict
        }

        if self.graph.num_latents > 0 and not enforce_acyclic_graph:
            metrics["confounders"] = self.get_confounder_metrics()
        return metrics

    @torch.no_grad()
    def get_confounder_metrics(self):
        """
        Returns metrics for detecting the latent confounders in the graph.
        """
        # Determine TP, FP, FN, and TN for latent confounder prediction
        l_score = self.get_confounder_scores()
        l_score = torch.triu(l_score, diagonal=1)
        l_predict = torch.stack(torch.where(l_score >= self.latent_threshold), dim=-1)
        l_predict = l_predict.cpu().numpy()
        match = (l_predict[:, None, :] == self.graph.latents[None, :, 1:]).all(axis=-1).any(axis=1).astype(np.int32)
        TP_latent = match.sum()
        FP_latent = (1 - match).sum()
        FN_latent = self.graph.num_latents - TP_latent
        num_pairs = self.num_vars*(self.num_vars-1)
        TN_latent = num_pairs - (TP_latent+FP_latent+FN_latent)

        metrics_conf = {
            "TP": int(TP_latent),
            "FP": int(FP_latent),
            "FN": int(FN_latent),
            "TN": int(TN_latent),
            "scores": l_score[self.graph.latents[:, 1], self.graph.latents[:, 2]].cpu().numpy().tolist()
        }
        return metrics_conf

    def get_state_dict(self):
        """
        Returns a dictionary of all important parameters to save the current prediction status.
        """
        state_dict = {
            "gamma": self.gamma.data.detach(),
            "theta": self.theta.data.detach(),
            "model": self.distribution_fitting_module.model.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads parameters from a state dictionary, obtained from 'get_state_dict'.
        """
        self.gamma.data = state_dict["gamma"]
        self.theta.data = state_dict["theta"]
        self.distribution_fitting_module.model.load_state_dict(state_dict["model"])

    def to(self, device):
        """
        Moves all PyTorch parameters to a specified device.
        """
        self.distribution_fitting_module.model.to(device)
        self.gamma.data = self.gamma.data.to(device)
        self.theta.data = self.theta.data.to(device)
        self.theta_optimizer.to(device)
        if hasattr(self.gamma_optimizer, "to"):
            self.gamma_optimizer.to(device)

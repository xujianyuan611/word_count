import matplotlib.pyplot as plt

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')

import torch
import numpy as np
from causal_graphs.graph_definition import CausalDAG  # Base class of causal graphs
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func  # Functions for generating new graphs
from causal_graphs.graph_visualization import visualize_graph  # Plotting the graph in matplotlib

graph = generate_categorical_graph(num_vars=8,
                                   min_categs=10,
                                   max_categs=10,
                                   graph_func=get_graph_func('random'),
                                   edge_prob=0.4,
                                   seed=42)

graph.save_to_file('hihi.pt')

print(graph)

visualize_graph(graph, figsize=(4, 4), show_plot=True)
# print(graph.sample(batch_size=10))
graph.sample(interventions={'C': np.array([0])})

# 上面是生成一个因果图的代码
# 下面是我们的ENCO算法的运行代码

from causal_discovery.enco import ENCO

# 为训练做准备工作
enco_module = ENCO(graph=graph)
if torch.cuda.is_available():
    enco_module.to(torch.device('cuda:0'))

# 开始训练
predicted_adj_matrix = enco_module.discover_graph(num_epochs=2)

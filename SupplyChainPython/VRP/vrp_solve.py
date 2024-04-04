
import os
import numpy as np
import torch
from VRP.creat_vrp import reward1

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from VRP.VRP_Actor import Model
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def discrete_cmap(N, base_cmap=None):
    # base = plt.cm.get_cmap(base_cmap)
    base = matplotlib.colormaps[base_cmap]

    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def vrp_solve(input_node, input_demand, Greedy=True):

    # node_ = np.loadtxt('./test_data/vrp100_test_data.csv', dtype=np.float32, delimiter=',')
    # demand_=np.loadtxt('./test_data/vrp100_demand.csv', dtype=np.float32, delimiter=',')
    # capcity_=np.loadtxt('./test_data/vrp100_capcity.csv', dtype=np.float32, delimiter=',')

    input_node = np.array(input_node)
    input_demand = np.array(input_demand)

    n_nodes = len(input_node)
    capcity = 5

    # Calculate the distance matrix
    edges = np.zeros((n_nodes, n_nodes, 1))
    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5
    for i, (x1, y1) in enumerate(input_node):
        for j, (x2, y2) in enumerate(input_node):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0] = d
    edges_ = edges.reshape(-1, 1)

    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    datas = []
    data = Data(x=torch.from_numpy(input_node).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges_).float(),
                demand=torch.tensor(input_demand).unsqueeze(-1).float(),
                capcity=torch.tensor(capcity).unsqueeze(-1).float())
    datas.append(data)

    data_loder = DataLoader(datas, batch_size=1)



    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)
    folder = 'VRP/trained'
    filepath = os.path.join(folder, '%s' % n_nodes)

    if os.path.exists(filepath):
        path1 = os.path.join(filepath, 'actor.pt')
        agent.load_state_dict(torch.load(path1, device))
    if Greedy:
        batch = next(iter(data_loder))
        batch.to(device)
        agent.eval()
        #-------------------------------------------------------------------------------------------Greedy
        with torch.no_grad():
            tour, _ = agent(batch, n_nodes * 2,True)
            #cost = reward1(batch.x, tour.detach(), n_nodes)
            #print(cost)
            #print(tour)
    #-------------------------------------------------------------------------------------------sampling1280
    else:
        datas_ = []
        batch_size1 = 128  # sampling batch_size
        for y in range(1280):
            data = Data(x=torch.from_numpy(input_node).float(), edge_index=edges_index,
                        edge_attr=torch.from_numpy(edges_).float(),
                        demand=torch.tensor(input_demand).unsqueeze(-1).float(),
                        capcity=torch.tensor(capcity).unsqueeze(-1).float())
            datas_.append(data)
        dl = DataLoader(datas_, batch_size=batch_size1)

        min_tour=[]
        min_cost=100
        T=1.2#Temperature hyperparameters
        for batch in dl:
            with torch.no_grad():
                batch.to(device)
                tour1, _ = agent(batch, n_nodes * 2,False, T)
                cost = reward1(batch.x, tour1.detach(), n_nodes)

                id = np.array(cost.cpu()).argmin()
                m_cost=np.array(cost.cpu()).min()
                tour1=tour1.reshape(batch_size1,-1)
                if m_cost<min_cost:
                    min_cost=m_cost
                    min_tour=tour1[id]

        tour=min_tour.unsqueeze(-2)

    print(tour)
    return tour[0].tolist()

#True:Greedy decoding / False:sampling1280
# vrp_matplotlib(Greedy=True)
import os
from typing import Optional

import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class NodewiseGenerator(nn.Module):
    def __init__(self, metadata, rand_dim, num_nodes, embed_dim, device, cat_class_num_list):
        super().__init__()

        self.metadata = metadata
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.device = device

        self.rand_to_proj = nn.Linear(rand_dim, (num_nodes - 1) * embed_dim)

        self.gcn = GCN(embed_dim, num_nodes, device, 0)

        final_dim = 0

        for n_i in range(num_nodes - 1):
            final_dim += metadata['details'][n_i]['n']

        self.proj_to_final = nn.Linear(num_nodes * embed_dim, final_dim)

        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, z, cond_tensor_, adj):
        '''
            z(torch.tensor)
            - shape: (batch_size, rand_dim)
        '''

        batch_size = z.shape[0]

        output_1 = self.leakyrelu(self.rand_to_proj(z))
        output_1_ = output_1.reshape(batch_size, self.num_nodes - 1, self.embed_dim)
        
        output_w_cond = torch.concat([output_1_, cond_tensor_], dim = 1)

        gnn_output = self.gcn(output_w_cond, adj) # (batch_size, num_nodes, embed_dim)
        gnn_output_ = gnn_output.reshape(batch_size, -1)

        n_output = self.proj_to_final(gnn_output_)

        return n_output


class Critic(nn.Module):
    def __init__(self, layers, dim_representation, device):
        super().__init__()

        self.layers = layers

        self.linear = nn.Linear(dim_representation, 1)
        self.device = device

    def forward(self, x, adj):
        '''
            output
            - shape: (batch, 1)
        '''

        batch_size = x.shape[0]

        x_representation = self.layers(x, adj)

        x_representation = x_representation.reshape(batch_size, -1)

        result = self.linear(x_representation)

        return result

class GCN(nn.Module):
    def __init__(self, embed_dim, num_nodes, device,dropout=0.1):
        super().__init__()

        self.num_layers = 5

        self.num_nodes = num_nodes

        self.gc1 = GraphConvolution(embed_dim, int(embed_dim * 2), num_nodes)
        self.gc2 = GraphConvolution(int(embed_dim * 2), int(embed_dim * 2), num_nodes)
        self.gc3 = GraphConvolution(int(embed_dim * 2), embed_dim, num_nodes)

        self.dropout_rate = 0
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.prelu = nn.PReLU()


    def forward(self, x, adj):
        x = x.reshape(x.shape[0], self.num_nodes, -1)

        x_init = x.clone()

        x_init_2 = x_init.repeat(1, 1, 2)

        x = self.prelu(self.gc1(x, adj) + x_init_2)
        x = self.prelu(self.gc2(x, adj) + x_init_2)
        x = self.prelu(self.gc3(x, adj) + x_init)

        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, bias = True):
        '''
            args:
                in_features(int): input feature dimension
                out_featuers(int): out feature dimension
                bias(boolean): if True, then Graph convolution operaion with bias
        '''
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias_tf = bias

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if self.bias_tf:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # stdv = 1. / (self.weight.size(1) ** (1/4))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias_tf:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input, adj):
        '''
            args:
                input(torch.tensor): input X
                - shape: (B, C, N, E_in==w_in)
                adj(torhc.tensor): adjacency matrix
                - shape: (C, N, N)
            output:
                torch.tensor
                - shape: (B, C, N, E_out==w_out)
        '''
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias_tf:
            return output + self.bias
        else:
            return output


class Classifier(nn.Module):
    def __init__(self, metadata, embed_dim, num_nodes, num_mode_list):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.metadata = metadata
        self.num_nodes = num_nodes
        self.num_columns = num_nodes - 1

        self.last_linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        for detail in metadata['details']:
            if detail['type'] == "category":
                layer_seq = nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=detail['n'])
                )
                self.last_linear_layers.append(layer_seq)
                self.activation_layers.append(nn.Sigmoid())
            else:
                # prediction
                layer_seq = nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=1)
                )
                self.last_linear_layers.append(layer_seq)
                self.activation_layers.append(None)

        self.relu = nn.ReLU()

    def forward(self, x, rand_idx):
        '''
        return:
            act_data(torch.tensor): classified output
            - shape: (batch_size, num_category)
        '''
        if self.metadata['details'][rand_idx]['type'] == 'category':
            act_data = torch.tanh(self.last_linear_layers[rand_idx](x[:, rand_idx]))
        else:
            act_data = torch.tanh(self.last_linear_layers[rand_idx](x[:, rand_idx]))

        return act_data
    
class Projection_virtual_node(nn.Module):
    '''
        Seperate weight matrices per each features.
    '''
    def __init__(self, input_dim, metadata, emb_dim, device):
        '''
            input_dim(int)
            metadata(dict)
            emb_dim(int)
            device
        '''
        super().__init__()
        self.metadata = metadata
        self.details = metadata['details']
        self.device = device
        self.emb_dim = emb_dim
        self.total_input_dim = input_dim

        self.column_wise_region = {}

        self.num_features = 0
        self.input_dim_list = []

        for col_name in self.metadata['columns']:
            detail = self.metadata['details'][col_name]
            type = detail["type"]

            if type == 'category':
                for cate_info in self.metadata['categori_embed_inform']:
                    if cate_info['col_name'] == col_name:
                        self.input_dim_list.append(cate_info['embed_dim'])
                        break
            else:
                self.input_dim_list.append(detail['n'])

            self.num_features = self.num_features + 1

        self.output_dim = self.num_features * emb_dim

        self.linear = nn.Linear(input_dim, self.output_dim)

        # define mask
        self.mask = torch.zeros(input_dim, self.output_dim)

        # row_idx: starting row idx to fill in 1 in self.mask
        row_idx = 0
        # col_idx: starting column idx to fill in 1 in self.mask
        col_idx = 0

        for in_dim in self.input_dim_list:
            self.mask[row_idx:row_idx+in_dim, col_idx:col_idx+self.emb_dim] = 1

            row_idx = row_idx + in_dim
            col_idx = col_idx + self.emb_dim

        self.mask = self.mask.T
        self.mask.requires_grad = False
        self.mask = self.mask.to(device)

        self.relu = nn.ReLU()

    def forward(self, x, cond_vec, cond_chosen_columns):
        '''
        args:
            x(tensor)
        '''
        real_weight = torch.mul(self.linear.weight, self.mask)
        real_weight = real_weight.to(self.device)

        bias = self.linear.bias
        bias.to(self.device)

        x_proj = torch.matmul(x, real_weight.T) + bias

        '''
            cond_proj
                shape: [batch_size, num_features * embed_size]
            cond_proj_reshape
                sahpe: [batch_size, num_features, embed_size]
            cond_proj_slice
                shape: [batch_size, embed_size]
        '''
        cond_proj = torch.matmul(cond_vec, real_weight.T) + bias
        cond_proj_reshape = tensor_reshape(cond_proj, self.metadata)
        cond_proj_slice = cond_proj_reshape[range(x.shape[0]), cond_chosen_columns[:, 0]]

        return x_proj, cond_proj_slice

    def reverse(self, proj):
        '''
        args:
            proj(tensor)
        '''
        real_weight = torch.mul(self.linear.weight, self.mask)
        real_weight.to(self.device)

        bias = self.linear.bias
        bias.to(self.device)

        proj_ = (proj - bias).unsqueeze(2)

        reverse_weight = torch.matmul(real_weight.transpose(0, 1),
                                      torch.linalg.pinv(torch.matmul(real_weight, real_weight.transpose(0, 1))))

        x = torch.matmul(reverse_weight, proj_).squeeze()

        return x

def tensor_reshape(input_tensor, metadata):
    '''
    Reshape tensor dimension
        from (num_samples, emb_dim*num_features)
        to (num_samples, num_features, emb_dim)
    '''
    num_features = metadata['num_features']
    output_tensor = input_tensor.reshape(input_tensor.shape[0], num_features, -1)

    return output_tensor

def tensor_recover(input_tensor):
    '''
    Recover tensor dimension
        from (num_samples, num_features, emb_dim)
        to (num_samples, emb_dim*num_features)
    '''
    output_tensor = input_tensor.reshape(input_tensor.shape[0], -1)

    return output_tensor
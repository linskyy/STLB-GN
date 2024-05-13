import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class GraphConv(nn.Module):
    r"""
    Dual Graph Convolution.

    Args:
        dim_in: input dimension.
        num_cheb_filter: output size.
        conv_type:gcn,cheb,jacobi.
        activation: default relu.
    """
    def __init__(self, in_dim, num_nodes, hidden_dim, emb_dim, conv_type=None, K=3,fusion=0):
        super(GraphConv, self).__init__()
        self.K = K
        self.fusion = fusion
        if conv_type == 'jacobi':
            self.a = nn.Parameter(torch.tensor(0.5), requires_grad=True)
            self.b = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(emb_dim, self.K, in_dim, hidden_dim),requires_grad=True)
        self.bias_pool = nn.Parameter(torch.FloatTensor(emb_dim, hidden_dim),requires_grad=True)
        self.hyperGNN_dim = 36
        self.middle_dim = 18
        self.embed_dim = emb_dim
        self.fc=nn.Sequential(
        OrderedDict([('fc1', nn.Linear(in_dim, self.hyperGNN_dim)),
                        ('sigmoid1', nn.Sigmoid()),
                        ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                        ('sigmoid2', nn.Sigmoid()),
                        ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
        self.w_conv = nn.Linear(in_dim * self.K, hidden_dim, bias=False)
        if not self.fusion:
            self.attn = nn.Parameter(torch.rand(num_nodes, hidden_dim), requires_grad=True)
            
        self.conv_type = conv_type
        nn.init.kaiming_uniform_(self.weights_pool, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias_pool, a=math.sqrt(5))


    def cheb_conv(self, x, node_embeddings):
        bs, _,num_nodes, _= x.size()

        supports1 = torch.eye(num_nodes).to(node_embeddings[0].device)
        filter = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter)) 
        adj_mx = GraphConv.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(3, 2))), supports1)
        h_list = [x, torch.matmul(adj_mx, x)]
        for _ in range(2, self.K):
            h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])

        h_p = torch.cat(h_list, dim=-1)
        h_n = torch.stack(h_list,dim=3)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)  
        bias = torch.matmul(node_embeddings[1], self.bias_pool)
        #regional-gcn
        h_n = torch.einsum('btnki,nkio->btno', h_n, weights) + bias     
        #pair-wise-gcn
        h_p = self.w_conv(h_p)
        if self.fusion == 1:
            h = h_n +h_p
        elif self.fusion == 2:
            h = self.attn * h_p + (1- self.attn) * h_n
        else:
            h = h_n

        return h


    def jacobi_conv(self, x, node_embeddings):
        bs, _,num_nodes, _= x.size()

        supports1 = torch.eye(num_nodes).to(node_embeddings[0].device)
        filter = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #[B,T,N,dim_in]
        adj_mx = GraphConv.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(3, 2))), supports1)
    
        h_list = [x, (self.a - self.b) / 2 * x + (self.a + self.b + 2) / 2 * torch.matmul(adj_mx, x)]
        for i in range(2, self.K):
            coef_l = 2 * i * (i + self.a + self.b) * (2 * i - 2 + self.a + self.b)
            coef_lm1_1 = (2 * i + self.a + self.b - 1) * (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 2)
            coef_lm1_2 = (2 * i + self.a + self.b - 1) * (self.a**2 - self.b**2)
            coef_lm2 = 2 * (i - 1 + self.a) * (i - 1 + self.b) * (2 * i + self.a + self.b)
            tmp1 =  coef_lm1_1 / coef_l
            tmp2 =  coef_lm1_2 / coef_l
            tmp3 =  coef_lm2 / coef_l
            nx = tmp1 * torch.matmul(adj_mx, h_list[i - 1]) + tmp2 * h_list[i - 1] - tmp3 * h_list[i - 2]
            norm = math.sqrt(pow(2, self.a + self.b + 1) * math.gamma(i + self.a + 1) *
                             math.gamma(i + self.b + 1 ) / ((2 * i + self.a + self.b) * 
                            math.gamma(i + self.a + self.b + 1) * math.factorial(i)))
            h_list.append(nx / norm)
        h_p = torch.cat(h_list, dim=-1)
        h_n = torch.stack(h_list,dim=3)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings[1], self.bias_pool)
        #regional-gcn
        h_n = torch.einsum('btnki,nkio->btno', h_n, weights) + bias     #b, N, dim_out
        #pair-wise-gcn
        h_p = self.w_conv(h_p)
        if self.fusion == 1:
            h = h_n +h_p
        elif self.fusion == 2:
            h = self.attn * h_p + (1- self.attn) * h_n
        else:
            h = h_n

        return h

    def forward(self, x, node_embeddings):
        return self.jacobi_conv(x, node_embeddings)
    
    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L



class GTU(nn.Module):
    def __init__(self, in_dim, kernel_size):
        super(GTU, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.in_channels = in_dim
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_dim, 2 * in_dim, kernel_size=(1, kernel_size), padding=(0, self.padding))
    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class STBlock(nn.Module):
    def __init__(self, in_dim, num_nodes, emb_dim, hidden_dim,  conv_type, fusion, K=3):
        
        super(STBlock, self).__init__()


        self.graph_conv = GraphConv(in_dim, num_nodes, hidden_dim, emb_dim, conv_type=conv_type, K=K,fusion=fusion)
        self.fusion = fusion
        self.gtu3 = GTU(hidden_dim, 3)
        self.gtu5 = GTU(hidden_dim, 5)
        self.gtu7 = GTU(hidden_dim, 7)
        if not self.fusion:
            self.t_aggr = nn.Sequential(
                nn.Linear(36, 12),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(12,12),
                nn.ReLU(inplace=True)
            )
        self.residual_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1)
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, node_embeddings):

        #dual graph conv
        h = self.graph_conv(x, node_embeddings)

        #multi-scale gated temporal conv
        if not self.fusion :
            x_gtu = []
            x_gtu.append(self.gtu3(h.transpose(1,3)))  # B,F,N,T-2
            x_gtu.append(self.gtu5(h.transpose(1,3)))  # B,F,N,T-4
            x_gtu.append(self.gtu7(h.transpose(1,3)))  # B,F,N,T-6
            time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-12
            h = self.t_aggr(time_conv).transpose(1,3)   # B,F,N,T->B,T,N,F
        else:
            h = (self.gtu3(h.transpose(1,3))+self.gtu5(h.transpose(1,3))+self.gtu7(h.transpose(1,3))).transpose(1,3) + h
        h_res = self.residual_conv(x.transpose(1, 3)).transpose(1, 3)
        h = torch.relu(h + h_res)
        return self.ln(h)


class STLBGN(nn.Module):
    def __init__(self, args):
        super(STLBGN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_units
        self.output_dim = args.output_dim
        self.emb_dim = args.embed_dim
        self.horizon = args.horizon
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.K = args.gcn_k
        self.fusion = args.fusion
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim)) 
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))   


        self.block_list = nn.ModuleList()


        self.block_list.append(STBlock(self.input_dim + self.emb_dim, self.num_node, self.emb_dim, self.hidden_dim, 'jacobi', fusion=self.fusion, K=self.K
        ))
        

        self.final_conv = nn.Conv2d(self.horizon, self.horizon, (1, self.hidden_dim))

    def forward(self, source):
        node_embedding1 = self.node_embeddings1 #N,emb_dim
        if self.use_D: #daily embedding
            t_i_d_data   = source[..., 1]

            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W: #weekly embedding
            d_i_w_data   = source[..., 2]

            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb) #B,T,N,emb_dim

        source = source[...,0].unsqueeze(-1)
        h = torch.cat((source,node_embedding1),dim=-1)
        h_list = []
        for net_block in self.block_list:
            h = net_block(h, [self.node_embeddings1, self.node_embeddings2])
            h_list.append(h)
        #skip connection
        h = torch.stack(h_list,dim=0).mean(dim=0)

        h = self.final_conv(h)
        return h
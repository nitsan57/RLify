import torch
import numpy as np
import torch.nn as nn
from .base_model import BaseModel


class DTREE(BaseModel):

    def __init__(
        self,
        input_shape: dict,
        out_shape: tuple,
        embed_dim: int = 128,
        depth: int = 2,
        activation: torch.nn.Module = nn.ReLU(),
    ):
        super().__init__(input_shape, out_shape)
        self.activation = activation

        self.input_size_dict = {
            k: np.prod(self.input_shape[k]) for k in self.input_shape
        }
        self.num_inputs = len(self.input_size_dict)

        self.l1 = torch.nn.ModuleDict(
            {
                k: nn.Sequential(nn.Linear(input_size, embed_dim), self.activation)
                for k, input_size in self.input_size_dict.items()
            }
        )
        self.embed_layer = torch.nn.ModuleDict(
            {
                k: nn.Sequential(
                    *[
                        nn.Sequential(nn.Linear(embed_dim, embed_dim), self.activation)
                        for i in range(depth)
                    ]
                )
                for k in self.input_size_dict
            }
        )
        self.concat_layer = DTree(
            embed_dim * self.num_inputs, embed_dim, depth=depth, activation=activation
        )
        self.out_layer = nn.Linear(embed_dim, np.prod(self.out_shape))

    def forward(self, x, d=None):  # d is for rnn api compability
        res_dict = dict()
        for k in x:
            layer_in = torch.flatten(x[k], start_dim=1)
            out = self.l1[k](layer_in)
            out = self.embed_layer[k](out)
            res_dict[k] = out

        res = torch.cat(list(res_dict.values()), 1)
        res = self.concat_layer(res)
        res = self.out_layer(res)
        return res

    def reset(self):
        pass


class DTreeNodes(torch.nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(
            DTreeNodes,
            self,
        ).__init__()
        flatten_shape = np.prod(input_dim).astype(np.int)
        all_right_params = torch.zeros(num_nodes, flatten_shape)
        for i in range(num_nodes):

            all_right_params[i] = torch.randn_like(all_right_params[i])  # right_params

        self.right_w = torch.nn.Parameter(all_right_params)

    def forward(self, x):
        b = x.shape[0]

        flatten_x = x.flatten(1)

        normlized_x = torch.nn.functional.normalize(flatten_x, dim=1)
        right = (
            torch.nn.functional.normalize(self.right_w, dim=1)
            .unsqueeze(0)
            .expand(b, self.right_w.shape[0], self.right_w.shape[1])
            .matmul(normlized_x.unsqueeze(-1))
        )

        right_dist = (1 - right) / 2

        left_dist = 1 - right_dist

        return right_dist, left_dist


class DTree(torch.nn.Module):
    def __init__(self, input_shape, out_shape, depth=2, activation=nn.ReLU()):
        super(DTree, self).__init__()
        self.is_rnn = False
        self.depth = depth
        self.num_nodes = 2 ** (depth + 1) - 1
        self.num_leaves = 2 ** (depth + 1)
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.activation = activation
        self.pre_network = nn.Sequential(
            nn.Linear(np.prod(input_shape), np.prod(input_shape) + 1),
            self.activation,
        )
        input_shape = np.prod(input_shape) + 1
        self.nodes = DTreeNodes(input_shape, num_nodes=self.num_nodes)
        self.output_dim = out_shape

        self.leaves_models = nn.Linear(
            np.prod(input_shape).astype(np.int), out_shape * self.num_leaves
        )  #

        self.leaves_route_indices = torch.zeros((self.num_leaves, self.depth + 1))
        self.leaves_route_branch_sides = torch.zeros((self.num_leaves, self.depth + 1))

        for i in range(self.num_leaves):
            temp = []
            temp_sides = []
            leaf_idx = self.num_nodes + i

            for j in range(self.depth):
                parent_idx = (leaf_idx - 1) // 2
                temp.append(parent_idx)
                if (leaf_idx - 1) % 2 == 0:
                    temp_sides.append(1)
                else:
                    temp_sides.append(0)
                leaf_idx = parent_idx

            if (leaf_idx - 1) % 2 == 0:
                temp_sides.append(1)
            else:
                temp_sides.append(0)

            temp.append(0)
            self.leaves_route_indices[i] = torch.tensor(temp[::-1]).int()
            self.leaves_route_branch_sides[i] = torch.tensor(temp_sides[::-1]).int()
            self.i = 0

    def forward(self, x):
        x = self.pre_network(x)
        return self.simple_leaves_parallel_forward(x)

    def simple_leaves_parallel_forward(self, x):
        b = x.shape[0]
        all_indices = torch.zeros((2, self.num_nodes, b), device=x.device).float()
        results = torch.zeros(b, self.num_leaves, self.output_dim, device=x.device)
        rights, lefts = self.nodes(x)
        all_indices[0, :, :] = rights.squeeze(-1).transpose(0, 1)
        all_indices[1, :, :] = lefts.squeeze(-1).transpose(0, 1)

        leaves_indices = torch.zeros(self.num_leaves, b, device=x.device)

        results = self.leaves_models(x.flatten(1)).reshape(
            b, self.num_leaves, self.output_dim
        )

        eps = 1e-2
        probs = all_indices[
            self.leaves_route_branch_sides.long(), self.leaves_route_indices.long()
        ]
        log_probs = torch.clamp(probs, eps, 1 - eps).log()
        leaves_indices = torch.sum(log_probs, 1)

        scaled_leaves_indices = leaves_indices.exp()
        entropy = -torch.sum(scaled_leaves_indices * leaves_indices, 0)
        max_entropy = -torch.sum(
            (torch.ones(self.num_leaves) / self.depth)
            * (torch.ones(self.num_leaves) / self.depth).log()
        )
        entropy_loss = entropy / max_entropy

        s_w = scaled_leaves_indices.transpose(0, 1)
        results = results * ((torch.randn(1) + 1) + entropy_loss.mean())
        # if self.i % 100 == 0:
        #     print(entropy_loss.mean().item())
        #     self.i = 0
        # self.i += 1

        return (results * s_w.unsqueeze(-1)).sum(1)

    def normal_forward(self, x):
        b = x.shape[0]
        all_indices = torch.zeros((2, self.num_nodes, b), device=x.device).float()

        results = torch.zeros(self.num_leaves, self.output_dim[-1], b, device=x.device)
        rights, lefts = self.nodes(x)
        all_indices[0, :, :] = rights.squeeze().transpose(0, 1)
        all_indices[1, :, :] = lefts.squeeze().transpose(0, 1)

        leaves_indices = torch.zeros(self.num_leaves, b, device=x.device)
        for i in range(self.num_leaves):
            model = self.leaves_models[i]
            results[i] = model(x).transpose(1, 0)

            probs = all_indices[
                self.leaves_route_branch_sides[i].tolist(),
                self.leaves_route_indices[i].tolist(),
            ]
            eps = 1e-2  # torch.finfo(all_indices.dtype).eps
            log_probs = torch.clamp(probs, eps, 1 - eps).log()
            leaves_indices[i] = torch.sum(log_probs, 0)  # *(4**self.depth)

        scaled_leaves_indices = leaves_indices * self.gamma  # .exp()*self.gamma

        s_w = torch.softmax(scaled_leaves_indices, 0).transpose(0, 1)

        results = results.permute(2, 0, 1)
        return torch.unsqueeze((results * s_w.unsqueeze(-1)).sum(1), dim=-1)

    def reset(self):
        pass


class DBoosTree(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_trees, depth):
        super(DBoosTree, self).__init__()
        self.trees = []
        input_shape = np.prod(input_shape)
        for i in range(num_trees):
            self.trees.append(DTREE(input_shape + i, output_shape, depth=depth))
        self.trees = torch.nn.ModuleList(self.trees)

    def forward(self, x):
        self.outputs = []
        x = x.flatten(1)
        for tree in self.trees:
            x = torch.cat([x, tree(x)])
            self.outputs.append(tree(x))
        out = torch.cumsum(self.outputs, dim=0)
        return out

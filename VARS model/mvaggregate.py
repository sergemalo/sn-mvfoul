from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2) # learnable transformation matrix, Dimension (E, E)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        # E = embedding dimension

        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)) # shape: (B, V, E)


        ##################### VIEW ATTENTION #####################

        # Transform output embeddings
        aux = torch.matmul(aux, self.attention_weights) # Dimension (B, V, E)

        # Compute dot product (similarity, "attention weights") between every transformed embedding
        aux_t = aux.permute(0, 2, 1) # Dimension (B, E, V)
        prod = torch.bmm(aux, aux_t) # Batch matrix-matrix product, Dimension (B, V, V)
        relu_res = self.relu(prod)   # Ensures non-negative "attention weights", Dimension (B, V, V)
        
        # Divide "attention weights" by sum across batch
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0) # Compute attention weights sum across batch                           
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0)) # Divide each element by batch sum
        final_attention_weights = final_attention_weights.T
        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V)) # Dimension (B, V, V)

        # Sum up all attention weights for a view
        final_attention_weights = torch.sum(final_attention_weights, dim=1) # Dimension (B, V)

        # Scale the embeddings by the attention weights
        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))    # aux.squeeze() (B, V, E), 
                                                                                    # final_attention_weights.unsqueeze(-1) (B, V, 1)
                                                                                    # element-wise multiplication

        # Aggregate over the views by summing the embeddings
        output = torch.sum(output, 1) # Dimension (B, E)
        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention

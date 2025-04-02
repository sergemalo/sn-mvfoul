from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim):
        super().__init__()
        self.model = model
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2) # learnable transformation matrix, Dimension (E, E)      
        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        # E = embedding dimension

        aux = unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True) # shape: (B, V, E)


        ##################### VIEW ATTENTION #####################

        # Transform output embeddings
        transformed_emb = torch.matmul(aux, self.attention_weights) # Dimension (B, V, E)

        # Compute view similarity (dot product of the transformed embeddings)
        transformed_emb_t = transformed_emb.permute(0, 2, 1) # Dimension (B, E, V)
        similarity_matrix = torch.bmm(transformed_emb, transformed_emb_t) # Batch matrix-matrix product, Dimension (B, V, V)
        similarity_matrix_relu = self.relu(similarity_matrix)   # Ensures non-negative "attention weights", Dimension (B, V, V)
        
        # Normalize the similarities (divide each element by sum across batch)
        similarity_sum_batch = torch.sum(torch.reshape(similarity_matrix_relu, (B, V*V)).T, dim=0).unsqueeze(0) # Compute attention weights sum across batch                           
        normalized_similarity = torch.div(torch.reshape(similarity_matrix_relu, (B, V*V)).T, similarity_sum_batch.squeeze(0)) # Divide each element by batch sum
        normalized_similarity = normalized_similarity.T
        normalized_similarity = torch.reshape(normalized_similarity, (B, V, V)) # Dimension (B, V, V)
        normalized_similarity = torch.sum(normalized_similarity, dim=1) # Dimension (B, V)

        # Scale the embeddings by the attention weights
        output = torch.mul(transformed_emb.squeeze(), normalized_similarity.unsqueeze(-1))      # transformed_emb.squeeze() (B, V, E), 
                                                                                                # normalized_similarity.unsqueeze(-1) (B, V, 1)
                                                                                                # element-wise multiplication

        # Aggregate over the views by summing the embeddings
        output = torch.sum(output, 1) # Dimension (B, E)
        return output.squeeze(), normalized_similarity


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400):
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
            self.aggregation_model = ViewMaxAggregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim)

    def forward(self, mvimages):

        pooled_view, attention = self.aggregation_model(mvimages)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention

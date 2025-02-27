from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self, vid_enc_model, feat_dim):
        super().__init__()
        self.vid_enc_model = vid_enc_model
        self.feature_dim = feat_dim

        # Initialize learnable transformation matrix
        r1 = -1
        r2 = 1
        self.transformation = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2) # initialize learnable transformation matrix, Dimension (E, E)      

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):

        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        # E = embedding dimension

        aux = unbatch_tensor(self.vid_enc_model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True) # shape: (B, V, E)


        ##################### VIEW ATTENTION #####################
        # Transform output embeddings
        aux = torch.matmul(aux, self.transformation) # Dimension (B, V, E)

        # Compute dot product (similarity) between every transformed embedding
        aux_t = aux.permute(0, 2, 1) # Dimension (B, E, V)
        similarities = torch.bmm(aux, aux_t) # Batch matrix-matrix product, Dimension (B, V, V)
        relu_similarities = self.relu(similarities)  # Ensures non-negative similarities, Dimension (B, V, V)
        
        # Divide similarities by sum across batch
        aux_sum = torch.sum(torch.reshape(relu_similarities, (B, V*V)).T, dim=0).unsqueeze(0) # Compute attention weights sum across batch                           
        attention_weights = torch.div(torch.reshape(relu_similarities, (B, V*V)).T, aux_sum.squeeze(0)) # Divide each element by batch sum
        attention_weights = attention_weights.T
        attention_weights = torch.reshape(attention_weights, (B, V, V)) # Dimension (B, V, V)

        # Sum up all attention weights for a view
        attention_weights = torch.sum(attention_weights, dim=1) # Dimension (B, V)

        # Scale the embeddings by the attention weights
        output = torch.mul(aux.squeeze(), attention_weights.unsqueeze(-1))      # aux.squeeze() (B, V, E), 
                                                                                # final_attention_weights.unsqueeze(-1) (B, V, 1)
                                                                                # element-wise multiplication

        # Aggregate over the views by summing the embeddings
        output = torch.sum(output, 1) # Dimension (B, E)
        return output.squeeze(), attention_weights



class ViewMaxAggregate(nn.Module):
    def __init__(self, vid_enc_model):
        super().__init__()
        self.vid_enc_model = vid_enc_model

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width

        # Get the video embeddings and apply maximum aggregation
        aux = unbatch_tensor(self.vid_enc_model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux



class ViewAvgAggregate(nn.Module):
    def __init__(self, vid_enc_model):
        super().__init__()
        self.vid_enc_model = vid_enc_model

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width

        # Get the video embeddings and apply average aggregation
        aux = unbatch_tensor(self.vid_enc_model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux
    

# TODO: Define a class that consists of a fully connected layer between aux and feat_dim



class MVAggregate(nn.Module):
    def __init__(self, vid_enc_model, agr_type="max", feat_dim=400):
        super().__init__()
        self.agr_type = agr_type

        # MLP between aggregated embeddings and classifiers
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        # MLP for offense severity (task 2)
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )

        # MLP for foul classification (task 1)
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        # Video encoder and aggregation
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(vid_enc_model=vid_enc_model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(vid_enc_model=vid_enc_model)
        else:
            self.aggregation_model = WeightedAggregate(vid_enc_model=vid_enc_model)


    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention

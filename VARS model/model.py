
import __future__
import torch
from mvaggregate import MVAggregate
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.video import swin3d_t, Swin3D_T_Weights



class MVNetwork(torch.nn.Module):

    def __init__(self, vid_enc_name='r2plus1d_18', agr_type='max'):
        super().__init__()

        self.vid_enc_name = vid_enc_name
        self.agr_type = agr_type
        
        self.feat_dim = 512

        # Argument checked in main, one condition will be matched
        if vid_enc_name == "r3d_18":                        # ResNet
            weights_model = R3D_18_Weights.DEFAULT          # KINETICS400_V1
            vid_enc_model = r3d_18(weights=weights_model)
        elif vid_enc_name == "r2plus1d_18":                 # R(2+1)D
            weights_model = R2Plus1D_18_Weights.DEFAULT     # KINETICS400_V1
            vid_enc_model = r2plus1d_18(weights=weights_model)
        elif vid_enc_name == "mvit_v2_s":                   # MViTv2 (Small)
            weights_model = MViT_V2_S_Weights.DEFAULT       # KINETICS400_V1
            vid_enc_model = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        elif vid_enc_name == "swin3d_t":                    # Swin3d Transformer (Tiny)
            weights_model = Swin3D_T_Weights.DEFAULT        # KINETICS400_V1
            vid_enc_model = swin3d_t(weights=weights_model)
                
        self.mvnetwork = MVAggregate(
            vid_enc_model=vid_enc_model,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)

import __future__
import torch
from mvaggregate import MVAggregate
from channel_reducer import ChannelReducer
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.models.video import swin3d_s, Swin3D_S_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights
from torchvision.models.video import s3d, S3D_Weights



class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18', agr_type='max', reduce_channels=False):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.feat_dim = 512

        if net_name == "r3d_18":                            # ResNet
            weights_model = R3D_18_Weights.DEFAULT          # KINETICS400_V1
            network = r3d_18(weights=weights_model)
        elif net_name == "r2plus1d_18":                     # R(2+1)D
            weights_model = R2Plus1D_18_Weights.DEFAULT     # KINETICS400_V1
            network = r2plus1d_18(weights=weights_model)
        elif net_name == "mvit_v2_s":                       # MViTv2 (small)
            weights_model = MViT_V2_S_Weights.DEFAULT       # KINETICS400_V1
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "swin3d_t":                        # Swin3d Transformer (tiny)
            weights_model = Swin3D_T_Weights.DEFAULT        # KINETICS400_V1
            network = swin3d_t(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "swin3d_s":                        # Swin3d Transformer (tiny)
            weights_model = Swin3D_S_Weights.DEFAULT        # KINETICS400_V1
            network = swin3d_s(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        
                
        network.fc = torch.nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
        )

        if reduce_channels:
            print("--> Adding channel reducer")
            self.channel_reducer = ChannelReducer(
                in_channels=4,
                out_channels=3
            )
        else:
            self.channel_reducer = None

    def forward(self, mvimages):
        # Expect input shape: (batch, views, channels, frames, height, width)
        print(f"MVNetwork received input shape: {mvimages.shape}")
        
        if self.channel_reducer:
            print(f"Applying channel reduction from {mvimages.shape[2]} to {self.channel_reducer.out_channels} channels")
            mvimages = self.channel_reducer(mvimages)
            print(f"After channel reduction: {mvimages.shape}")
            
            # Verify the tensor has the expected 6D shape after channel reduction
            if len(mvimages.shape) != 6:
                raise ValueError(f"Expected 6D tensor after channel reduction, but got shape {mvimages.shape}")
        
        return self.mvnetwork(mvimages)

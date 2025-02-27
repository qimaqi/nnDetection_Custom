# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import re

from .videomae import video_vit
from .videomae.video_vit import LayerNorm3d
from .videomae.map_to_decoder import Decoder24_Upsampler, Decoder17_Upsampler, Decoder11_Upsampler, Decoder5_Upsampler,SingleConv3DBlock, Decoder_Map

import torch.nn.functional as F
import numpy as np 
import omegaconf


def interpolate_pretrained_pos_enc_encoder(args: dict, state_dict: dict, seg_temporal_pos=False) -> dict:
    """
    Adjusts the pretrained positional encoding tensor to fit the current model's dimensions.(larger)

    Args:
        args (dict): The input arguments to the model
        state_dict (dict): The loaded state dictionary to adjust

    Returns:
        dict: The adjusted state dictionary with the updated positional encoding
    """

    if 'patch_embed.proj.weight' in state_dict.keys():
        kernal_shape= state_dict['patch_embed.proj.weight'].shape[-1]
    else:
        kernal_shape=16

    orig_patches_per_dim = args.checkpoint_shape // kernal_shape  # original 224x224 model with patch size 16
    new_patches_per_dim = args.img_size // args.patch_size
    if orig_patches_per_dim != new_patches_per_dim:
        if not seg_temporal_pos:
            # we add a small number (0.1) to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            #prepare for interpolation
            h0, w0 = new_patches_per_dim + 0.1, new_patches_per_dim + 0.1
            # print("pos_enc before interpolate",  state_dict["pos_embed_spatial"].size()) # ([1, 196, 1024]) #224/16=14, 14*14=196, 
                                                                                             #[1,64 (8*8), 1024]
            pos_enc = state_dict["pos_embed_spatial"].reshape(
                1, orig_patches_per_dim, orig_patches_per_dim, -1
            )
            print("pos_enc before interpolate", pos_enc.size())
            dim = pos_enc.shape[-1]
            pos_enc = pos_enc.permute(0, 3, 1, 2)
            pos_enc = torch.nn.functional.interpolate(
                pos_enc,
                scale_factor=(h0 / orig_patches_per_dim, w0 / orig_patches_per_dim),
                mode="bicubic",
                align_corners=False,
            )
            assert int(h0) == pos_enc.shape[-2] and int(w0) == pos_enc.shape[-1]
            pos_enc = pos_enc.permute(0, 2, 3, 1).view(1, -1, dim)
            print("pos_enc after interpolate", pos_enc.size())
            state_dict["pos_embed_spatial"] = pos_enc
        else:
            raise NotImplementedError

    # check pos_embed_temporal
    orig_pos_embed_temporal_dim = 8
    new_pos_embed_temporal_dim = args.num_frames // args.t_patch_size
    if orig_pos_embed_temporal_dim != new_pos_embed_temporal_dim:
        pos_enc = state_dict["pos_embed_temporal"].reshape(
                1, orig_pos_embed_temporal_dim, -1
            )
        print("pos_enc temporal before interpolate", pos_enc.size())
        dim = pos_enc.shape[-1]
        pos_enc = pos_enc.permute(0, 2, 1)
        pos_enc = torch.nn.functional.interpolate(
            pos_enc,
            size=(new_pos_embed_temporal_dim,),
            mode="linear",
            align_corners=False,
        )
        assert new_pos_embed_temporal_dim == pos_enc.shape[-1], pos_enc.shape
        pos_enc = pos_enc.permute(0, 2, 1).view(1, -1, dim)
        print("pos_enc temporal after interpolate", pos_enc.size())
        state_dict["pos_embed_temporal"] = pos_enc

        expected_embed_dim = args.embed_dim  # e.g., 1024
        if "pos_embed_temporal" in state_dict:
            current_dim = state_dict["pos_embed_temporal"].shape[-1]
            if current_dim != expected_embed_dim:
                print(f"Adjusting pos_embed_temporal channel dim from {current_dim} to {expected_embed_dim}")
                # Here, state_dict["pos_embed_temporal"] has shape [1, T, current_dim] (e.g., [1, 6, 768]).
                # Since we want to interpolate the last dimension (the embedding dimension),
                # we can directly call F.interpolate, which expects an input of shape (N, C, L) for 1D interpolation.
                # Our tensor is already [1, T, current_dim] with T as the channel dimension if we interpret it as (N, C, L)?
                # Actually, in our case, T is the temporal tokens (6) and current_dim is the embedding dimension (768).
                # F.interpolate expects (N, C, L), so if we treat T as channels and current_dim as length, we would get [1,6,1024] if we interpolate length.
                # However, we want to change the last dimension, so we treat the tensor as is.
                pos_embed = state_dict["pos_embed_temporal"]  # shape: [1, T, current_dim]
                # Directly interpolate along the last dimension:
                pos_embed = F.interpolate(pos_embed, size=expected_embed_dim, mode="linear", align_corners=False)
                # This will transform [1, 6, 768] to [1, 6, 1024]
                state_dict["pos_embed_temporal"] = pos_embed
                print("pos_embed_temporal after channel adjustment", pos_embed.size())
            
    return state_dict

def adjust_state_dict_keys(state_dict: dict) -> dict:
    """
    Adjust the keys of the state dict to match the model.

    Args:
        state_dict (dict): The state dict to adjust

    Returns:
        dict: The adjusted state dict
    """
    if "pred_head.transforms.0.4.weight" not in state_dict:
        return state_dict
    adjusted_state_dict = {}
    adjusted_state_dict["decoder_norm.weight"] = state_dict.pop(
        "pred_head.transforms.0.4.weight"
    )
    adjusted_state_dict["decoder_norm.bias"] = state_dict.pop(
        "pred_head.transforms.0.4.bias"
    )
    # if args.model.pred_t_dim == 8:
    #     adjusted_state_dict["decoder_pred.weight"] = state_dict.pop(
    #         "pred_head.projections.0.weight"
    #     )
    #     adjusted_state_dict["decoder_pred.bias"] = state_dict.pop(
    #         "pred_head.projections.0.bias"
    #     )
        
    for key in state_dict.keys():
        adjusted_state_dict[
            key.replace("pred_head.transforms.0", "decoder_blocks")
        ] = state_dict[key]


    return adjusted_state_dict

# def adjust_pos_embed_temporal_channels(state_dict, expected_dim):
#     """
#     Upscale pos_embed_temporal from its current channel dimension to expected_dim.
#     Assumes state_dict["pos_embed_temporal"] has shape [1, T, C].
#     """
#     if "pos_embed_temporal" in state_dict:
#         pos_embed = state_dict["pos_embed_temporal"]
#         current_dim = pos_embed.shape[-1]
#         if current_dim != expected_dim:
#             print(f"Adjusting pos_embed_temporal from {pos_embed.shape} to (1, T, {expected_dim})")
#             # Permute to [1, C, T] so we can interpolate along the channel dimension.
#             pos_embed = pos_embed.permute(0, 2, 1)
#             # Interpolate along the channel dimension.
#             pos_embed = F.interpolate(pos_embed, size=expected_dim, mode="linear", align_corners=False)
#             print(, pos_embed.size()))
#             # Permute back to [1, T, expected_dim]
#             #pos_embed = pos_embed.permute(0, 2, 1)
#             state_dict["pos_embed_temporal"] = pos_embed
#     return state_dict

def load_pretrained_weights_encoder(model, model_cfg):
    saved_model = torch.load(model_cfg['pretrained_path'], map_location="cpu")
    # print("path ",model_cfg['pretrained_path'])
    # print("keys", saved_model.keys())
    # model.load_state_dict(saved_model['model_state'])
    # # Load the optimizer state
    # optimizer.load_state_dict(saved_model['optimizer_states'])

    # # Verify the optimizer state
    # print("Optimizer state loaded successfully:")
    # print(optimizer.state_dict())
    
    # raise ValueError("verify optimizer state")

    if 'model_state' in saved_model.keys():
        pretrained_dict = saved_model['model_state']
    elif 'model' in saved_model.keys():
        pretrained_dict = saved_model['model']
    else:
        raise ValueError("Could not find the model state in the loaded model")
    
    # print("Pretrained model keys:")
    # for key in pretrained_dict.keys():
    #     print(key)

    # if is mae or hiera encoder  
    pretrained_dict = adjust_state_dict_keys(pretrained_dict)

    # print("Pretrained model keys:")
    # for key in pretrained_dict.keys():
    #     print(key)

    pretrained_dict["decoder_pos_embed"] = pretrained_dict["decoder_pos_embed"][:, 1:, :]
    # check if we need to interpoalte the positional encoding
    # input size 
    if model_cfg['t_patch_size']!= 2 or model_cfg['patch_size'] != 16:
        # remove patch_embed from pretrained_dict
        pretrained_dict.pop("patch_embed.proj.weight")
        pretrained_dict.pop("patch_embed.proj.bias")

    if model_cfg['img_size'] != 224 or model_cfg['num_frames'] != 16:
        
        checkpoint_name = model_cfg['pretrained_path']
        #print("checkpoint_name", checkpoint_name)
        match = re.search(r'_(\d+)\.pth$', checkpoint_name)  # Matches the last number before ".pth"
        #print("match", match)
        if match:
            checkpoint_shape = int(match.group(1))  # Extract the first dimension (128 in this case)
        else:
            raise ValueError("Checkpoint file name does not contain the expected shape format.")
        #print("checkpoint_shape", checkpoint_shape)

        args = {'img_size': model_cfg['img_size'], 'num_frames': model_cfg['num_frames'], 't_patch_size': model_cfg['t_patch_size'], 'patch_size': model_cfg['patch_size'], 'checkpoint_shape': checkpoint_shape, 'embed_dim':model_cfg['embed_dim']}
        args = omegaconf.OmegaConf.create(args)
        
        print("args",args)
        pretrained_dict = interpolate_pretrained_pos_enc_encoder(args, pretrained_dict)

      #print("interpolated pretrained model keys:")
    # print("Pretrained model keys:")
    # for key in pretrained_dict.keys():
    #     print(key)

    # print("\nModel keys:")
    # for key in model.state_dict().keys():
    #     print(key)
    #exit()
    missing, unexpected = model.load_state_dict(
        pretrained_dict, strict=False
    )

    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    print("################### Done ###################")


class VideoMAE_Encoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        feature_shapes,
        img_size=224,
        patch_size=16,
        num_frames=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        t_patch_size=2,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        sep_pos_embed_decoder=False,
        trunc_init=False,
        cls_embed=False,
        use_lora=0, # 0 for not use lora
        upsample_func='conv',
        upsample_stage='direct',
        skip_connection=False,
        output_layers=[5,11,17,23],
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        cls_embed_decoder = False 
        self.cls_embed_decoder = cls_embed_decoder
        self.output_layers = output_layers
        
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path

        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.sep_pos_embed_decoder = sep_pos_embed_decoder
        self.cls_embed = cls_embed
        # 2 * 8 // 16
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        self.patch_info = None

        self.t_pred_patch_size = t_patch_size
        self.feature_shapes = feature_shapes

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.output_size = [num_frames, img_size, img_size]
        self.token_shape = [num_frames // t_patch_size, img_size // patch_size, img_size // patch_size]
        self.embed_dim = embed_dim

        self.skip_connection = skip_connection

        self.intermediate_feat_layer = output_layers


        # we need stage 0 1 2 3 4, stage 1 from 5, stage 2 from 11, stage 3 from 17, stage 4 from last
 
        
        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )

            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    use_lora=use_lora,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )
        
        # if self.map_to_decoder_type == 'conv':
        if self.skip_connection:
            self.decoder0_upsampler = Decoder_Map(in_planes = in_chans, out_planes=feature_shapes[0][0], input_size=feature_shapes[0][1:], output_size=feature_shapes[0][1:],
                                                    upsample_func=upsample_func, upsample_stage=upsample_stage)
        else:
            print("skip connection is false")
            print("set layer", 0, 'in plane', embed_dim, 'out plane', feature_shapes[0][0], 'input size', self.token_shape, 'output size', feature_shapes[0][1:])
            setattr
            #next line of code was added by 
            #self.channel_adjust = nn.Conv3d(embed_dim, feature_shapes[0][0], kernel_size=1) #

            self.decoder0_upsampler = Decoder_Map(in_planes=embed_dim,
                                             out_planes=feature_shapes[0][0],
                                             input_size=self.token_shape,
                                             output_size=feature_shapes[0][1:],
                                             upsample_func=upsample_func,
                                             upsample_stage=upsample_stage)
            
        

        assert len(self.intermediate_feat_layer) == len(feature_shapes) - 1, f"len(self.intermediate_feat_layer) {len(self.intermediate_feat_layer)} != len(feature_shapes) {len(feature_shapes)} - 1 "
        for num_j, layer_j in  enumerate(self.intermediate_feat_layer):
            print("set layer", layer_j, 'in plane', embed_dim, 'out plane', feature_shapes[num_j+1][0], 'input size', feature_shapes[num_j+1][1:], 'output size', feature_shapes[num_j+1][1:])
            setattr(self, f"decoder{layer_j}_upsampler", Decoder_Map(in_planes = embed_dim, 
                                                                        out_planes=feature_shapes[num_j+1][0], 
                                                                        input_size=self.token_shape, 
                                                                        output_size=feature_shapes[num_j+1][1:],
                                                                        upsample_func=upsample_func, upsample_stage=upsample_stage),
                                                                        ) 
        # raise ValueError("Not implemented yet")

        # if self.map_to_decoder_type == 'conv':
        #     self.decoder24_upsampler = Decoder24_Upsampler(embed_dim, 320)
        #     # Decoder_Map(in_planes = embed_dim, out_planes=320, input_size=) #
        #     # Decoder24_Upsampler(embed_dim, 320) # 

        #     operation_dict = {}
        #     if len(output_layers) == 3:
        #         self.decoder3_upsampler = Decoder17_Upsampler(embed_dim, 256)

        #         self.decoder2_upsampler = Decoder11_Upsampler(embed_dim, 128)
                
        #         self.decoder1_upsampler = Decoder5_Upsampler(embed_dim, 64)

        #         self.decoder0_upsampler = SingleConv3DBlock(in_chans, 32, kernel_size=3)
        #         # stage 0
        #         # x before torch.Size([4, 1, 16, 224, 224])
        #         # x after torch.Size([4, 32, 16, 224, 224])
        #         # stage 1
        #         # x before torch.Size([4, 32, 16, 224, 224])
        #         # x after torch.Size([4, 64, 8, 112, 112])
        #         # stage 2
        #         # x before torch.Size([4, 64, 8, 112, 112])
        #         # x after torch.Size([4, 128, 4, 56, 56])
        #         # stage 3
        #         # x before torch.Size([4, 128, 4, 56, 56])
        #         # x after torch.Size([4, 256, 4, 28, 28])
        #         # stage 4
        #         # x before torch.Size([4, 256, 4, 28, 28])
        #         # x after torch.Size([4, 320, 4, 14, 14])

        #         operation_dict['17'] = self.decoder3_upsampler
        #         operation_dict['11'] = self.decoder2_upsampler
        #         operation_dict['5'] = self.decoder1_upsampler
                 

        #     elif len(output_layers) == 4:
        #         # target 
        #         # stage 0
        #         # x before torch.Size([4, 1, 64, 128, 128])
        #         # x after torch.Size([4, 32, 64, 128, 128])
        #         # stage 1
        #         # x before torch.Size([4, 32, 64, 128, 128])
        #         # x after torch.Size([4, 64, 32, 64, 64])
        #         # stage 2
        #         # x before torch.Size([4, 64, 32, 64, 64])
        #         # x after torch.Size([4, 128, 16, 32, 32])
        #         # stage 3
        #         # x before torch.Size([4, 128, 16, 32, 32])
        #         # x after torch.Size([4, 256, 8, 16, 16])
        #         # stage 4
        #         # x before torch.Size([4, 256, 8, 16, 16])
        #         # x after torch.Size([4, 320, 4, 8, 8])
        #         # stage 5
        #         # x before torch.Size([4, 320, 4, 8, 8])
        #         # x after torch.Size([4, 320, 4, 4, 4])
        #         # print("input shape", x.shape)
        #         self.decoder4_upsampler = Decoder24_Upsampler(embed_dim, 320)
        #         operation_dict['19'] = self.decoder4_upsampler
        #         operation_dict['15'] = self.decoder3_upsampler
        #         operation_dict['10'] = self.decoder2_upsampler
        #         operation_dict['5'] = self.decoder1_upsampler
        #     self.operation_dict = operation_dict

        # else:
        #     raise NotImplementedError


        self.initialize_weights()
        print("model initialized")
        self.use_lora = use_lora



    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_size #self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * C))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, -1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, -1, T, H, W))
        return imgs

    def convert_3d_to_2d_tensor(self, x):
        N,C, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4) # B, 1024, 8, 14, 14 
        x = x.reshape([N*D, C, H, W])
        return x

    def convert_2d_to_3d_tensor(self, x, N):
        ND, C, H, W = x.size()
        D = ND // N 
        x = x.reshape([N, D, C, H, W])
        x = x.permute(0, 2, 1, 3, 4)
        return x


    def forward_encoder(self, x):
        # ([2, 3, 16, 224, 224])
        #  B, C, T, H, W 
        # print("encoder sample_x", x.size())
        multi_scale_feat = []
        if self.skip_connection:
            feat_0 = self.decoder0_upsampler(x)
            # print("map to decoder 0", feat_0.size())
            multi_scale_feat.append(feat_0) # do not need repeated color
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C) # combine temporal and spatial together
       
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        pos_embed = pos_embed.to(x.device)
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for layer_i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer_i in self.intermediate_feat_layer:
                intermediate_feat = self.convert_to_3d_tensor(x)

                # print("layer_i", layer_i, intermediate_feat.size())
                if self.skip_connection:
                    intermediate_feat = getattr(self, f"decoder{layer_i}_upsampler")(intermediate_feat)
                    # print("map to decoder", intermediate_feat.size())
                    multi_scale_feat.append(intermediate_feat)
        
        if not self.skip_connection:
            # use the last layer feature as shown in vitdet
            #print("debug x", x.size())
            intermediate_feat = self.convert_to_3d_tensor(x)
            #print("intermediate_feat", intermediate_feat.size())

            feat0 = self.decoder0_upsampler(intermediate_feat)
            #print("map to decoder 0", feat0.size())
            multi_scale_feat.append(feat0)
            #print("map to decoder 0", feat0.size())
            for num_j, layer_j in  enumerate(self.intermediate_feat_layer):
                intermediate_feat = self.convert_to_3d_tensor(x)
                # print(f"Before decoder {layer_j}, shape: {intermediate_feat.size()}") 
                intermediate_feat = getattr(self, f"decoder{layer_j}_upsampler")(intermediate_feat)
                # print("last layer map to decoder", num_j, intermediate_feat.size())
                multi_scale_feat.append(intermediate_feat)
            # 96 example size
                # map to decoder 5 torch.Size([1, 64, 32, 96, 96])
                # map to decoder 10 torch.Size([1, 128, 16, 48, 48])
                # map to decoder 15 torch.Size([1, 256, 8, 24, 24])
                # map to decoder 19 torch.Size([1, 320, 8, 12, 12])
                # map to decoder 23 torch.Size([1, 320, 8, 6, 6])
                        
        return multi_scale_feat

    def convert_to_3d_tensor(self, x):

        N = x.shape[0]
        C = x.shape[-1]
        # print("x size", x.size())
        x = x.view([N, self.input_size[0],self.input_size[1], self.input_size[2], C]) # B, 8, 14, 14, 512
        x = x.permute(0, 4, 1, 2, 3) # B, 1024, 8, 14, 14 
        return x

 
    def forward(self, imgs):
        input_dim = imgs.shape[1]
        if input_dim == 1:
            imgs = imgs.repeat(1,3,1,1,1)
        # print("====================================")
        # print("imgs", imgs.size())
        _ = self.patchify(imgs)
        multi_scale_feat = self.forward_encoder(imgs)
        return multi_scale_feat




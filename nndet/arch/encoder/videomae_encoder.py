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

from .videomae import video_vit
from .videomae.video_vit import LayerNorm3d
from .videomae.map_to_decoder import Decoder24_Upsampler, Decoder17_Upsampler, Decoder11_Upsampler, Decoder5_Upsampler,SingleConv3DBlock

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
    orig_patches_per_dim = 224 // 16  # original 224x224 model with patch size 16
    new_patches_per_dim = args.img_size // 16
    if orig_patches_per_dim != new_patches_per_dim:
        if not seg_temporal_pos:
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            h0, w0 = new_patches_per_dim + 0.1, new_patches_per_dim + 0.1
            # print("pos_enc before interpolate",  state_dict["pos_embed_spatial"].size()) # ([1, 196, 1024])
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

def load_pretrained_weights_encoder(model, model_cfg):
    saved_model = torch.load(model_cfg['pretrained_path'], map_location="cpu")
    
    if 'model_state' in saved_model.keys():
        pretrained_dict = saved_model['model_state']
    elif 'model' in saved_model.keys():
        pretrained_dict = saved_model['model']
    else:
        raise ValueError("Could not find the model state in the loaded model")
    # if is mae or hiera encoder  
    pretrained_dict = adjust_state_dict_keys(pretrained_dict)

    pretrained_dict["decoder_pos_embed"] = pretrained_dict["decoder_pos_embed"][:, 1:, :]
    # check if we need to interpoalte the positional encoding
    # input size 
    if model_cfg['img_size'] != 224 or model_cfg['num_frames'] != 16:
        args = {'img_size': model_cfg['img_size'], 'num_frames': model_cfg['num_frames'], 't_patch_size': 2}
        args = omegaconf.OmegaConf.create(args)
        pretrained_dict = interpolate_pretrained_pos_enc_encoder(args, pretrained_dict)


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
        map_to_decoder_type='conv',
    ):
        super().__init__()
        cls_embed_decoder = False 
        self.cls_embed_decoder = cls_embed_decoder

        
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.sep_pos_embed_decoder = sep_pos_embed_decoder
        self.cls_embed = cls_embed
        # 2 * 8 // 16
        self.t_patch_size = t_patch_size
        self.patch_size = patch_size
        self.patch_info = None
        self.map_to_decoder_type = map_to_decoder_type
        self.t_pred_patch_size = t_patch_size

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
        self.output_size = [num_frames,img_size, img_size]
        self.embed_dim = embed_dim

        self.intermediate_feat_layer = [5,11,17]
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
                )
                for i in range(depth)
            ]
        )

        if self.map_to_decoder_type == 'conv':
            self.decoder24_upsampler = Decoder24_Upsampler(embed_dim, 320) # TODO change 320 from config
            # encoder dimension: patch size 16, t_patch size 2 
            # each encoder dim: 8x14x14x1024 
            self.decoder17_upsampler = Decoder17_Upsampler(embed_dim, 256)

            self.decoder11_upsampler = Decoder11_Upsampler(embed_dim, 128)
            
            self.decoder5_upsampler = Decoder5_Upsampler(embed_dim, 64)

            self.decoder0_upsampler = SingleConv3DBlock(in_chans, 32, kernel_size=3)

        else:
            raise NotImplementedError


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
        feat_0 = self.decoder0_upsampler(x)
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
                if layer_i == 5:
                    intermediate_feat = self.decoder5_upsampler(intermediate_feat)
                elif layer_i == 11:
                    intermediate_feat = self.decoder11_upsampler(intermediate_feat)
                elif layer_i == 17:
                    intermediate_feat = self.decoder17_upsampler(intermediate_feat)
                multi_scale_feat.append(intermediate_feat)


        intermediate_feat = self.convert_to_3d_tensor(x)
        intermediate_feat = self.decoder24_upsampler(intermediate_feat)
        multi_scale_feat.append(intermediate_feat)
        
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




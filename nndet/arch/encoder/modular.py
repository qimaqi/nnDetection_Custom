"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Dict, Optional

from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock
from nndet.arch.encoder.videomae_encoder import VideoMAE_Encoder, load_pretrained_weights_encoder
from nndet.arch.encoder.swinunetr import SwinUNETR_Encoder
import numpy as np
import os 

__all__ = ["Encoder", "VideoMAEEncoder", "SwinUnetrEncoder"]


def calculate_3dconv_output_shape(input_shape, conv_kernels, strides, start_channels=1, max_channels=320):
    # input_shape: (B, F, H, W, Z) => tuple (B, F, H, W, Z)
    F, H, W, Z = input_shape
    
    output_shapes = [(start_channels, H, W, Z)]
    F_last = start_channels
    # Loop through each layer's kernel and stride
    # print("conv_kernels", conv_kernels)
    # print("strides", strides)
    for kernel, stride in zip(conv_kernels, strides):
        # Compute the output height, width, and depth using the formula
        # TODO
        out_H = (H ) // stride[0] 
        out_W = (W ) // stride[1] 
        out_Z = (Z ) // stride[2]
        
        F_last = min(max_channels, F_last*2)
        # Store the output shape for the current layer
        output_shapes.append((F_last, out_H, out_W, out_Z))
        
        
        # Update the dimensions for the next layer
        H, W, Z = out_H, out_W, out_Z
    
    return output_shapes


def dummy_calculate_output_channel(conv_kernels, in_channels, plan_cfg, model_cfg):
    """
    Dummy function to calculate output channels
    """
    out_channels = []

    stages_num = len(conv_kernels)
    for i in range(stages_num):
        out_channels.append(1)

    return 1


class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 ):
        """
        Build a modular encoder model with specified blocks
        The Encoder consists of "stages" which (in general) represent one
        resolution in the resolution pyramid. The first level alwasys has
        full resolution.

        Args:
            conv: conv generator to use for internal convolutions
            strides: strides for pooling layers. Should have one
                element less than conv_kernels
            conv_kernels: kernel sizes for convolutions
            block_cls: generate a block of convolutions (
                e.g. stacked residual blocks)
            in_channels: number of input channels
            start_channels: number of start channels
            stage_kwargs: additional keyword arguments for stages.
                Defaults to None.
            out_stages: define which stages should be returned. If `None` all
                stages will be returned.Defaults to None.
            first_block_cls: generate a block of convolutions for the first stage
                By default this equal the provided block_cls
        """
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        stages = []
        self.out_channels = []
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_channels = _block.get_output_channels()
            self.out_channels.append(in_channels)
            stages.append(_block)
        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward data through encoder
        
        Args:
            x: input data
        
        Returns:
            List[torch.Tensor]: list of output from stages defined by
                param:`out_stages`
        """
        outputs = []
        # for 64x128x128 input
        # stage 0
        # x before torch.Size([4, 1, 64, 128, 128])
        # x after torch.Size([4, 32, 64, 128, 128])
        # stage 1
        # x before torch.Size([4, 32, 64, 128, 128])
        # x after torch.Size([4, 64, 32, 64, 64])
        # stage 2
        # x before torch.Size([4, 64, 32, 64, 64])
        # x after torch.Size([4, 128, 16, 32, 32])
        # stage 3
        # x before torch.Size([4, 128, 16, 32, 32])
        # x after torch.Size([4, 256, 8, 16, 16])
        # stage 4
        # x before torch.Size([4, 256, 8, 16, 16])
        # x after torch.Size([4, 320, 4, 8, 8])
        # stage 5
        # x before torch.Size([4, 320, 4, 8, 8])
        # x after torch.Size([4, 320, 4, 4, 4])
        # print("input shape", x.shape)
        #  [(32, 64, 128, 128), (64, 32, 64, 64), (128, 16, 32, 32), (256, 8, 16, 16), (320, 4, 8, 8), (320, 4, 4, 4)]
        

        # for 96x192x192 input:
        # x before torch.Size([4, 1, 96, 192, 192])
        # x after torch.Size([4, 32, 96, 192, 192])
        # stage 1
        # x before torch.Size([4, 32, 96, 192, 192])
        # x after torch.Size([4, 64, 48, 96, 96])
        # stage 2
        # x before torch.Size([4, 64, 48, 96, 96])
        # x after torch.Size([4, 128, 24, 48, 48])
        # stage 3
        # x before torch.Size([4, 128, 24, 48, 48])
        # x after torch.Size([4, 256, 12, 24, 24])
        # stage 4
        # x before torch.Size([4, 256, 12, 24, 24])
        # x after torch.Size([4, 320, 6, 12, 12])
        # stage 5
        # x before torch.Size([4, 320, 6, 12, 12])
        # x after torch.Size([4, 320, 6, 6, 6])
        for stage_id, module in enumerate(self.stages):
            # print("stage", stage_id)
            # print("x before", x.shape)
            x = module(x)
            # print("x after", x.shape)
            if stage_id in self.out_stages:
                outputs.append(x)
        return outputs

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map inside the forward pass

        Returns
            list: list with number of channels corresponding to returned feature maps
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        Compute number backbone strides for 2d and 3d case and all options of network

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * pool_size for prev_stride, pool_size
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides

class VideoMAEEncoder(AbstractEncoder):
    def __init__(self,
                #  conv: Callable[[], nn.Module],
                 dim: int,
                 model_cfg: Dict,
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                #  block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                #  stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                #  first_block_cls: Optional[AbstractBlock] = None,
                 ):
        """
        Build a modular encoder model with specified blocks
        The Encoder consists of "stages" which (in general) represent one
        resolution in the resolution pyramid. The first level alwasys has
        full resolution.

        Args:
            conv: conv generator to use for internal convolutions
            strides: strides for pooling layers. Should have one
                element less than conv_kernels
            conv_kernels: kernel sizes for convolutions
            block_cls: generate a block of convolutions (
                e.g. stacked residual blocks)
            in_channels: number of input channels
            start_channels: number of start channels
            stage_kwargs: additional keyword arguments for stages.
                Defaults to None.
            out_stages: define which stages should be returned. If `None` all
                stages will be returned.Defaults to None.
            first_block_cls: generate a block of convolutions for the first stage
                By default this equal the provided block_cls
        """
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = dim 
        print("====================================")
        print("model_cfg", model_cfg)

        # if stage_kwargs is None:
        #     stage_kwargs = [{}] * self.num_stages
        # elif isinstance(stage_kwargs, dict):
        #     stage_kwargs = [stage_kwargs] * self.num_stages
        # assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        # if first_block_cls is None:
        #     first_block_cls = block_cls

        stages = []
        self.out_channels = []
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides

        #### add videomae encoder here #####
        """init some parameters"""
        """some other parameters can change from model config"""
        encoder_cfg = model_cfg["encoder_kwargs"]
        feature_shapes = calculate_3dconv_output_shape([self.dim, encoder_cfg["num_frames"],encoder_cfg["img_size"] ,encoder_cfg["img_size"] ],
                                                       conv_kernels, strides, start_channels, max_channels)
                                                       
        print("feature_shapes", feature_shapes)
        feature_shapes = np.array(feature_shapes)

        self.stages = VideoMAE_Encoder(
            img_size=encoder_cfg["img_size"], 
            patch_size=encoder_cfg["patch_size"],
            num_frames=encoder_cfg["num_frames"],
            output_layers=encoder_cfg["output_layers"],
            feature_shapes=feature_shapes,
            t_patch_size=encoder_cfg["t_patch_size"],
            upsample_func=encoder_cfg["upsample_func"],
            upsample_stage=encoder_cfg["upsample_stage"],
            skip_connection=encoder_cfg["skip_connection"],
            use_lora=encoder_cfg.get("use_lora",0),
            drop=encoder_cfg.get("drop", 0),
            attn_drop=encoder_cfg.get("attn_drop", 0),
            drop_path=encoder_cfg.get("drop_path", 0),
        )

        if encoder_cfg['pretrained_path'] is not None and os.path.exists(encoder_cfg['pretrained_path']):
            load_pretrained_weights_encoder(self.stages, encoder_cfg)

        self.out_channels = feature_shapes[:, 0].tolist()
        # if len(conv_kernels) == 5: 
        #     # using 16x224x224 input
        #     self.out_channels = [32, 64, 128, 256, 320] # TODO
        # elif len(conv_kernels) == 6:
        #     # using 64x128x128 input
        #     self.out_channels = [32, 64, 128, 256, 320, 320]

        # for stage_id in range(self.num_stages):
        #     if stage_id == 0:
        #         _block = first_block_cls(
        #             conv=conv,
        #             in_channels=in_channels,
        #             out_channels=start_channels,
        #             conv_kernel=conv_kernels[stage_id],
        #             stride=None,
        #             max_out_channels=max_channels,
        #             **stage_kwargs[stage_id],
        #         )
        #     else:
        #         _block = block_cls(
        #             conv=conv,
        #             in_channels=in_channels,
        #             out_channels=None,
        #             conv_kernel=conv_kernels[stage_id],
        #             stride=strides[stage_id - 1],
        #             max_out_channels=max_channels,
        #             **stage_kwargs[stage_id],
        #         )
        #     in_channels = _block.get_output_channels()
        #     self.out_channels.append(in_channels)
        #     stages.append(_block)
        # self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward data through encoder
        
        Args:
            x: input data
        
        Returns:
            List[torch.Tensor]: list of output from stages defined by
                param:`out_stages`
        """
        outputs = []
        # stage 0
        # x before torch.Size([4, 1, 16, 224, 224])
        # x after torch.Size([4, 32, 16, 224, 224])
        # stage 1
        # x before torch.Size([4, 32, 16, 224, 224])
        # x after torch.Size([4, 64, 8, 112, 112])
        # stage 2
        # x before torch.Size([4, 64, 8, 112, 112])
        # x after torch.Size([4, 128, 4, 56, 56])
        # stage 3
        # x before torch.Size([4, 128, 4, 56, 56])
        # x after torch.Size([4, 256, 4, 28, 28])
        # stage 4
        # x before torch.Size([4, 256, 4, 28, 28])
        # x after torch.Size([4, 320, 4, 14, 14])

        all_features = self.stages(x)
        for layer_i, output_i in enumerate(all_features):
            # print(f"output_{layer_i} shape", output_i.shape)
            if layer_i in self.out_stages:
                outputs.append(output_i)

        return outputs

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map inside the forward pass

        Returns
            list: list with number of channels corresponding to returned feature maps
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        Compute number backbone strides for 2d and 3d case and all options of network

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * pool_size for prev_stride, pool_size
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides


class SwinUnetrEncoder(AbstractEncoder):
    def __init__(self,
                #  conv: Callable[[], nn.Module],
                 dim: int,
                 model_cfg: Dict,
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                #  block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                #  stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                #  first_block_cls: Optional[AbstractBlock] = None,
                 ):

        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = dim 
        print("====================================")
        print("model_cfg", model_cfg)

        # if stage_kwargs is None:
        #     stage_kwargs = [{}] * self.num_stages
        # elif isinstance(stage_kwargs, dict):
        #     stage_kwargs = [stage_kwargs] * self.num_stages
        # assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        # if first_block_cls is None:
        #     first_block_cls = block_cls

        stages = []
        self.out_channels = []
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides

        #### add videomae encoder here #####
        """init some parameters"""
        """some other parameters can change from model config"""
        encoder_cfg = model_cfg["encoder_kwargs"]

        feature_shapes = calculate_3dconv_output_shape([self.dim, encoder_cfg["img_size"][0], encoder_cfg["img_size"][1], encoder_cfg["img_size"][2]],
                                                       conv_kernels, strides, start_channels, max_channels)
                                                       
        print("feature_shapes", feature_shapes)
        feature_shapes = np.array(feature_shapes)

        self.stages = SwinUNETR_Encoder(
            img_size=encoder_cfg["img_size"], 
            in_channels=in_channels,
            depths=encoder_cfg["depths"],
            num_heads=encoder_cfg["num_heads"],
            feature_size=encoder_cfg["feature_size"],
            map_to_decoder_type=encoder_cfg["map_to_decoder_type"],
            feature_shapes=feature_shapes,
            skip_connection=encoder_cfg["skip_connection"],
        )

        if encoder_cfg['pretrained_path'] is not None:
            # load_pretrained_weights_encoder_swinunetr(self.stages, encoder_cfg)
            weights = torch.load(encoder_cfg['pretrained_path'])
            print("weights", weights['state_dict'].keys())
            self.stages.load_from(weights)

        
        self.out_channels = feature_shapes[:, 0].tolist()

        # for stage_id in range(self.num_stages):
        #     if stage_id == 0:
        #         _block = first_block_cls(
        #             conv=conv,
        #             in_channels=in_channels,
        #             out_channels=start_channels,
        #             conv_kernel=conv_kernels[stage_id],
        #             stride=None,
        #             max_out_channels=max_channels,
        #             **stage_kwargs[stage_id],
        #         )
        #     else:
        #         _block = block_cls(
        #             conv=conv,
        #             in_channels=in_channels,
        #             out_channels=None,
        #             conv_kernel=conv_kernels[stage_id],
        #             stride=strides[stage_id - 1],
        #             max_out_channels=max_channels,
        #             **stage_kwargs[stage_id],
        #         )
        #     in_channels = _block.get_output_channels()
        #     self.out_channels.append(in_channels)
        #     stages.append(_block)
        # self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward data through encoder
        
        Args:
            x: input data
        
        Returns:
            List[torch.Tensor]: list of output from stages defined by
                param:`out_stages`
        """
        outputs = []
        # stage 0
        # x before torch.Size([4, 1, 64, 128, 128])
        # x after torch.Size([4, 32, 64, 128, 128])
        # stage 1
        # x before torch.Size([4, 32, 64, 128, 128])
        # x after torch.Size([4, 64, 32, 64, 64])
        # stage 2
        # x before torch.Size([4, 64, 32, 64, 64])
        # x after torch.Size([4, 128, 16, 32, 32])
        # stage 3
        # x before torch.Size([4, 128, 16, 32, 32])
        # x after torch.Size([4, 256, 8, 16, 16])
        # stage 4
        # x before torch.Size([4, 256, 8, 16, 16])
        # x after torch.Size([4, 320, 4, 8, 8])
        # stage 5
        # x before torch.Size([4, 320, 4, 8, 8])
        # x after torch.Size([4, 320, 4, 4, 4])
        # print("input shape", x.shape)

        # enc0 torch.Size([1, 48, 64, 128, 128])
        # enc1 torch.Size([1, 48, 32, 64, 64])
        # enc2 torch.Size([1, 96, 16, 32, 32])
        # enc3 torch.Size([1, 192, 8, 16, 16])
        # dec4 torch.Size([1, 768, 2, 4, 4])

        # hidden_states_out 5
        # enc0 torch.Size([1, 48, 64, 128, 128])
        # hidden_states_out[0] torch.Size([1, 48, 32, 64, 64])
        # hidden_states_out[1] torch.Size([1, 96, 16, 32, 32])
        # hidden_states_out[2] torch.Size([1, 192, 8, 16, 16])
        # hidden_states_out[3] torch.Size([1, 384, 4, 8, 8])
        # hidden_states_out[4] torch.Size([1, 768, 2, 4, 4])
        all_features = self.stages(x)
        for layer_i, output_i in enumerate(all_features):
            # print(f"output_{layer_i} shape", output_i.shape)
            if layer_i in self.out_stages:
                outputs.append(output_i)

        return outputs

    def get_channels(self) -> List[int]:
        """
        Compute number of channels for each returned feature map inside the forward pass

        Returns
            list: list with number of channels corresponding to returned feature maps
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        Compute number backbone strides for 2d and 3d case and all options of network

        Returns
            List[List[int]]: defines the absolute stride for each output
                feature map with respect to input size
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * pool_size for prev_stride, pool_size
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides

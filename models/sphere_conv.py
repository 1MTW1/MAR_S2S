from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class SphereConv2d(nn.Conv2d):
    """수평(경도) 방향은 circular padding, 수직(위도) 방향은 극(pole) 반전 reflection 을
    적용하는 2D 합성곱 (전지구 구면 경계 처리)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode=None,
        padding_value=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

        # assert padding == 1, "For now, SphereConv2d only tested on padding=1 for spherical convolution."
        # self.padding = padding
        # assert self.kernel_size[0] == self.kernel_size[1] == 3, \
        # "SphereConv2d currently only tested on 3x3 kernels for spherical convolution."
        assert self.stride[0] == self.stride[1] == 1, (
            "SphereConv2d currently only tested on stride=1 for spherical convolution. "
        )

        if padding_mode is not None:
            Warning(
                f"This module has special padding modes, the passed {padding_mode} will be ignored."
            )
        if padding_value is not None:
            Warning(
                f"This module has special padding modes, the passed {padding_value} will be ignored."
            )

    @staticmethod
    def sphere_pad(input: torch.Tensor, padding: Tuple[int] = (1, 1)) -> torch.Tensor:
        """구면 합성곱용 4D 텐서(B,C,H,W) 패딩.
        경도(width)는 circular, 위도(height)는 극 처리(roll+flip)로 패딩한다.

        Args:
            input: (B, C, H, W) 입력 텐서
            padding: 각 변의 패딩 수 (padH, padW)
        Returns:
            구면 경계조건이 적용된 패딩 텐서
        """
        assert input.dim() == 4, (
            "Input tensor must be 4D (batch, channels, height, width)"
        )
        assert input.shape[3] % 2 == 0, (
            "Width of the input tensor must be even for proper shperical padding"
        )
        half_width = input.shape[3] // 2

        top_rows = input[:, :, : padding[0], :]
        top_rows = torch.roll(top_rows, shifts=half_width, dims=3)
        top_rows = torch.flip(top_rows, dims=[2])
        bottom_rows = input[:, :, -padding[0] :, :]
        bottom_rows = torch.roll(bottom_rows, shifts=half_width, dims=3)
        bottom_rows = torch.flip(bottom_rows, dims=[2])
        input = torch.cat([top_rows, input, bottom_rows], dim=2)

        return F.pad(input, (padding[1], padding[1], 0, 0), mode="circular")

    def top_conv(self, input: torch.Tensor) -> torch.Tensor:
        """패딩 후 입력의 상단 슬라이스에 합성곱 적용 (구면 합성곱의 최상단 행 처리).
        원본은 self.weight.data 를 in-place flip→복원 했는데, 이는 leaf 텐서의 version 을
        올려 backward 를 깨뜨린다(추론만 가능). 미분 가능하도록 flip 한 clone 사용."""
        p = self.padding[0]
        kernel = self.weight.clone()
        kernel[:, :, :p, :] = torch.flip(kernel[:, :, :p, :], dims=[3])
        return F.conv2d(input, kernel, self.bias, self.stride, 0, self.dilation, self.groups)

    def bottom_conv(self, input: torch.Tensor) -> torch.Tensor:
        """패딩 후 입력의 하단 슬라이스에 합성곱 적용 (구면 합성곱의 최하단 행 처리).
        top_conv 와 동일하게 in-place 미분 문제를 피하기 위해 flip 한 clone 사용."""
        p = self.padding[0]
        kernel = self.weight.clone()
        kernel[:, :, -p:, :] = torch.flip(kernel[:, :, -p:, :], dims=[3])
        return F.conv2d(input, kernel, self.bias, self.stride, 0, self.dilation, self.groups)

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        raise NotImplementedError(
            " SphereConv2d does not support _conv_forward method. Use forward method instead."
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (B, C, H, W)
        example:
        tmp = torch.arange(0, 24).view(1, 1, 3, 8)
        conv_cls = SphereConv2d(1, 1, 5, 1, 5//2)
        print(tmp)
        print(conv_cls.sphere_pad(tmp, (5//2, 5//2)))
        >>>
        tensor([[[[ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23]]]])
        tensor([[[[10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13],
                [ 2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5],
                [ 6,  7,  0,  1,  2,  3,  4,  5,  6,  7,  0,  1],
                [14, 15,  8,  9, 10, 11, 12, 13, 14, 15,  8,  9],
                [22, 23, 16, 17, 18, 19, 20, 21, 22, 23, 16, 17],
                [18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21],
                [10, 11, 12, 13, 14, 15,  8,  9, 10, 11, 12, 13]]]])

        conv_cls.weight.data = torch.tensor([[[[0,1,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0]]]], requires_grad=True, dtype=torch.float32)
        conv_cls.bias.data = torch.tensor([0.0], requires_grad=True, dtype=torch.float32)
        print(conv_cls.weight.data.shape)
        >>>
        tensor([[[[0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 0.]]]])

        conv_cls(tmp.float())
        >>>
        tensor([[[[44., 48., 52., 40., 44., 48., 52., 40.],
                [48., 44., 48., 44., 48., 44., 48., 44.],
                [52., 40., 44., 48., 52., 40., 44., 48.]]]], grad_fn=<CatBackward0>)
        """
        input = self.sphere_pad(input, padding=self.padding)
        top_slice = input[:, :, : self.kernel_size[0], :]
        mid_slice = input[:, :, self.stride[0] : -self.stride[0], :]
        bottom_slice = input[:, :, -self.kernel_size[0] :, :]
        top_slice = self.top_conv(top_slice)
        # print("top slice", top_slice, top_slice.shape)
        mid_slice = F.conv2d(
            mid_slice,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
        # print("mid slice", mid_slice, mid_slice.shape)
        bottom_slice = self.bottom_conv(bottom_slice)
        # print("bottom slice", bottom_slice, bottom_slice.shape)
        return torch.cat([top_slice, mid_slice, bottom_slice], dim=2)

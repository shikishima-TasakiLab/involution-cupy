from typing import Any, Optional, Tuple, Union
from collections import namedtuple
from string import Template

import torch
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import HalfTensor, FloatTensor, DoubleTensor

import cupy

__all__ = ['Involution2d']

_stream = namedtuple('Stream', ['ptr'])

CUDA_NUM_THREADS = 512
# CUDA_NUM_THREADS = 1024

_kernel_loop: str = '''
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)
'''

_involution2d_kernel: str = _kernel_loop + '''
extern "C"
__global__ void involution2d_forward_kernel(
const ${dtype}* bottom_data, const ${dtype}* weight_data, ${dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_height}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_width}; ++kw) {
        const int h_in = -${pad_height} + h * ${stride_height} + kh * ${dilation_height};
        const int w_in = -${pad_width} + w * ${stride_width} + kw * ${dilation_width};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_height} + kh) * ${kernel_width} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_involution2d_kernel_backward_grad_input: str = _kernel_loop + '''
extern "C"
__global__ void involution2d_backward_grad_input_kernel(
    const ${dtype}* const top_diff, const ${dtype}* const weight_data, ${dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_height}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_width}; ++kw) {
        const int h_out_s = h + ${pad_height} - kh * ${dilation_height};
        const int w_out_s = w + ${pad_width} - kw * ${dilation_width};
        if (((h_out_s % ${stride_height}) == 0) && ((w_out_s % ${stride_width}) == 0)) {
          const int h_out = h_out_s / ${stride_height};
          const int w_out = w_out_s / ${stride_width};
          if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_height} + kh) * ${kernel_width} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_involution2d_kernel_backward_grad_weight: str = _kernel_loop + '''
extern "C"
__global__ void involution2d_backward_grad_weight_kernel(
    const ${dtype}* const top_diff, const ${dtype}* const bottom_data, ${dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_width} / ${top_height} / ${top_width}) % ${kernel_height};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_width};
    const int h_in = -${pad_height} + h * ${stride_height} + kh * ${dilation_height};
    const int w_in = -${pad_width} + w * ${stride_width} + kw * ${dilation_width};
    if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_height} / ${kernel_width} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_height} / ${kernel_width} / ${top_height} / ${top_width}) % ${batch_size};
      ${dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h) * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


def _dtype(t) -> str:
    if isinstance(t, FloatTensor):
        return 'float'
    elif isinstance(t, DoubleTensor):
        return 'double'
    elif isinstance(t, HalfTensor):
        return 'half'


@cupy._util.memoize(for_each_device=True)
def _load_kernel(kernel_name: str, code: str, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


def _get_blocks(n: int) -> int:
    return (n - 1) // CUDA_NUM_THREADS + 1


class _involution2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight: torch.Tensor, stride: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int]) -> torch.Tensor:
        assert input.ndim == 4 and input.is_cuda
        assert weight.ndim == 6 and weight.is_cuda
        assert isinstance(stride, tuple) and len(stride) == 2
        assert isinstance(padding, tuple) and len(padding) == 2
        assert isinstance(dilation, tuple) and len(dilation) == 2
        batch_size, channels, height, width = input.shape
        kernel_height, kernel_width = weight.shape[2:4]
        out_height = (
            height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
        out_width = (
            width + 2 * padding[1] - dilation[0] * (kernel_width - 1) - 1) // stride[1] + 1

        output: torch.Tensor = input.new(
            batch_size, channels, out_height, out_width)
        num_elements: int = output.numel()

        with torch.cuda.device_of(input):
            kernel = _load_kernel('involution2d_forward_kernel', _involution2d_kernel, dtype=_dtype(input), nthreads=num_elements,
                                  batch_size=int(batch_size), channels=int(channels), groups=int(weight.shape[1]),
                                  bottom_height=int(height), bottom_width=int(width), top_height=int(out_height), top_width=int(out_width),
                                  kernel_height=int(kernel_height), kernel_width=int(kernel_width),
                                  stride_height=stride[0], stride_width=stride[1], dilation_height=dilation[0], dilation_width=dilation[1],
                                  pad_height=padding[0], pad_width=padding[1])
            kernel(block=(CUDA_NUM_THREADS, 1, 1),
                   grid=(_get_blocks(num_elements), 1, 1),
                   args=[input.data_ptr(), weight.data_ptr(),
                         output.data_ptr()],
                   stream=_stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.shape
        kernel_height, kernel_width = weight.shape[2:4]
        out_height, out_width = grad_output.shape[2:]

        grad_input, grad_weight = None, None

        opt = dict(dtype=_dtype(grad_output), batch_size=int(batch_size), channels=int(channels), groups=int(weight.shape[1]),
                   bottom_height=int(height), bottom_width=int(width), top_height=int(out_height), top_width=int(out_width),
                   kernel_height=int(kernel_height), kernel_width=int(kernel_width), stride_height=stride[0], stride_width=stride[1],
                   dilation_height=dilation[0], dilation_width=dilation[1], pad_height=padding[0], pad_width=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input: torch.Tensor = input.new(input.shape)

                num_elements: int = grad_input.numel()
                opt['nthreads'] = num_elements

                kernel = _load_kernel('involution2d_backward_grad_input_kernel',
                                      _involution2d_kernel_backward_grad_input, **opt)
                kernel(block=(CUDA_NUM_THREADS, 1, 1),
                       grid=(_get_blocks(num_elements), 1, 1),
                       args=[grad_output.data_ptr(), weight.data_ptr(),
                             grad_input.data_ptr()],
                       stream=_stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight: torch.Tensor = weight.new(weight.shape)

                num_elements = grad_weight.numel()
                opt['nthreads'] = num_elements

                kernel = _load_kernel('involution2d_backward_grad_weight_kernel',
                                      _involution2d_kernel_backward_grad_weight, **opt)
                kernel(block=(CUDA_NUM_THREADS, 1, 1),
                       grid=(_get_blocks(num_elements), 1, 1),
                       args=[grad_output.data_ptr(), input.data_ptr(),
                             grad_weight.data_ptr()],
                       stream=_stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _involution2d_cuda(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
        stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1) -> torch.Tensor:

    assert input.shape[0] == weight.shape[0]
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    assert input.shape[-2] // stride[0] == weight.shape[-2]
    assert input.shape[-1] // stride[1] == weight.shape[-1]
    if input.is_cuda:
        out: torch.Tensor = _involution2d.apply(
            input, weight, stride, padding, dilation)
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class Involution2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 7,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 3,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 sigma_mapping: Optional[nn.Module] = None,
                 reduce_ratio: int = 1,
                 ) -> None:
        """2D Involution: https://arxiv.org/pdf/2103.06255.pdf

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (Union[int, Tuple[int, int]], optional): Kernel size to be used. Defaults to 7.
            stride (Union[int, Tuple[int, int]], optional): Stride factor to be utilized. Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): Padding to be used in unfold operation. Defaults to 3.
            dilation (Union[int, Tuple[int, int]], optional): Dilation in unfold to be employed. Defaults to 1.
            groups (int, optional): Number of groups to be employed. Defaults to 1.
            bias (bool, optional): If true bias is utilized in each convolution layer. Defaults to False.
            sigma_mapping (Optional[nn.Module], optional): Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
            reduce_ratio (int, optional): Reduce ration of involution channels. Defaults to 1.
        """
        super(Involution2d, self).__init__()
        assert isinstance(in_channels, int) and in_channels > 0, \
            '"in_channels" must be a positive integer.'
        assert isinstance(out_channels, int) and out_channels > 0, \
            '"out_channels" must be a positive integer.'
        assert isinstance(kernel_size, (int, tuple)), \
            '"kernel_size" must be an int or a tuple of ints.'
        assert isinstance(stride, (int, tuple)), \
            '"stride" must be an int or a tuple of ints.'
        assert isinstance(padding, (int, tuple)), \
            '"padding" must be an int or a tuple of ints.'
        assert isinstance(dilation, (int, tuple)), \
            '"dilation" must be an int or a tuple of ints.'
        assert isinstance(groups, int) and groups > 0, \
            '"groups" must be a positive integer.'
        assert in_channels % groups == 0, '"in_channels" must be divisible by "groups".'
        assert out_channels % groups == 0, '"out_channels" must be divisible by "groups".'
        assert isinstance(bias, bool), '"bias" must be a bool.'
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            '"sigma_mapping" muse be an int or a tuple of ints.'
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, \
            '"reduce_ratio" must be a positive integer.'

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Tuple[int, int] = _pair(kernel_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)
        self.dilation: Tuple[int, int] = _pair(dilation)
        self.groups: int = groups
        self.bias: bool = bias
        self.reduce_ratio: int = reduce_ratio

        self.sigma_mapping = sigma_mapping if isinstance(sigma_mapping, nn.Module) else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels //
                           self.reduce_ratio, momentum=0.3),
            nn.ReLU()
        )
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=bias) \
            if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(
            kernel_size=self.stride) if self.stride[0] > 1 or self.stride[1] > 1 else nn.Identity()
        self.reduce_mapping = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels // self.reduce_ratio, kernel_size=1, bias=bias)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups, kernel_size=1, bias=bias)

    def __repr__(self) -> str:
        """Method returns information about the module

        Returns:
            str: Info string
        """
        return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size=({self.kernel_size[0]}, {self.kernel_size[1]}), '
            f'stride=({self.stride[0]}, {self.stride[1]}), padding=({self.padding[0]}, {self.padding[1]}), dilation=({self.dilation[0], self.dilation[1]}), '
            f'groups={self.groups}, bias={self.bias}, reduce_ratio={self.reduce_ratio}, sigma_mapping={str(self.sigma_mapping)}'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            input (torch.Tensor): Input tensor of the shape [batch size, in channels, height, width]

        Returns:
            torch.Tensor: Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        assert input.ndim == 4, f'Input tensor to Involution2d must be 4d but {input.ndim}d tensor is given.'

        kernel: torch.Tensor = self.span_mapping(
            self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        batch_size, _, height, width = kernel.shape

        input_init: torch.Tensor = self.initial_mapping(input)

        if input.is_cuda:
            kernel: torch.Tensor = kernel.view(
                batch_size, self.groups, self.kernel_size[0], self.kernel_size[1], height, width)
            return _involution2d_cuda(input_init, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            in_height, in_width = input.shape[-2:]
            out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (
                self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (
                self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

            input_unfolded: torch.Tensor = F.unfold(
                input_init, self.kernel_size, self.dilation, self.padding, self.stride)
            input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels //
                                                 self.groups, self.kernel_size[0] * self.kernel_size[1], out_height, out_width)

            kernel = kernel.view(
                batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)

            output: torch.Tensor = (kernel * input_unfolded).sum(dim=3)
            return output.view(batch_size, -1, output.shape[-2], output.shape[-1])

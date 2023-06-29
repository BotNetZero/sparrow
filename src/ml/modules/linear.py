# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/28
Description   : NN module with Quantization
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from src.common.symbol import DataType
from src.ml.modules.base import BaseModuleFP8


class LinearFP8(BaseModuleFP8):
	"""
	revise nn.Linear
	storage dtype: uint8, compute dtype: fp16
	"""
	def __init__(self, in_features: int, out_features: int, bias: bool = True, format="e4m3", device=None) -> None:
		super().__init__(format)
		dtype = DataType.fp8.value
		factory_kwargs = {'device': device, 'dtype': dtype}

		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(			# for storage, no grad
			torch.zeros((out_features, in_features), **factory_kwargs),
			requires_grad=False)
		if bias:
			self.bias = nn.Parameter(		# for storage, no grad
				torch.zeros(out_features, **factory_kwargs),
				requires_grad=False)	#
		else:
			self.register_parameter('bias', None)
		#
		self.init_parameters()

	def init_parameters(self):
		"""
		Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
		uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
		https://github.com/pytorch/pytorch/issues/57109
		"""
		_weight = nn.Parameter(
			torch.zeros_like(self.weight, device=self.weight.device, dtype=torch.float16),
			requires_grad=False
		)
		_bias = nn.Parameter(
				torch.zeros_like(self.bias, device=self.bias.device, dtype=torch.float16),
				requires_grad=False)	#
		nn.init.kaiming_uniform_(_weight, a=math.sqrt(5))
		self.weight = self.fp8formatter.convert_to_fp8(_weight)

		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(_weight)
			bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
			nn.init.uniform_(_bias, -bound, bound)
			self.bias = self.fp8formatter.convert_to_fp8(_bias)

	def forward(self, input: Tensor) -> Tensor:
		# return F.linear(input, self.weight, self.bias)
		pass


	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


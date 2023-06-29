# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/29
Description   :
"""
from typing import Any
import torch
import torch.nn as nn
from torch.autograd.function import Function
from src.utils.fp8 import FP8E4M3


class QntFP8(Function):
	"""
	FP8 --> FP16
	"""
	@staticmethod
	def forward(ctx, tensor_fp8):
		pass

	@staticmethod
	def backward(ctx, grad_output):
		pass



class DQntFP8(Function):
	"""
	FP16 --> FP8
	"""
	pass

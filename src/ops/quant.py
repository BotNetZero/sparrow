# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   : quantization
"""
from torch.autograd.function import Function


def convert_to_fp(tensor, target_type):
	pass


def convert_to_integer(tensor, target_type):
	pass


class Quant(Function):
	@staticmethod
	def forward(ctx, tensor):
		pass

	@staticmethod
	def backward(ctx, grad):
		pass

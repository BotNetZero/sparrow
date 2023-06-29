# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   : Linear operation
"""
from torch.autograd.function import Function


class _Linear(Function):
	"""

	"""
	@staticmethod
	def forward(ctx, inp, weight, bias):
		pass


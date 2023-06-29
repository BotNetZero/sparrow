# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/29
Description   : NN module for FP8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.symbol import DataType, FP8Format


class BaseModuleFP8(nn.Module):
	def __init__(self, format="e4m3") -> None:
		super().__init__()
		format = format.lower()
		if format not in (FP8Format.e4m3.name, FP8Format.e5m2.name):
			raise ValueError(f"not support FP8 format: {format}")
		self.fp8formatter = FP8Format[format].value


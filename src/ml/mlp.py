# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/29
Description   :
"""
import torch
import torch.nn as nn
from src.ml.modules.linear import LinearFP8


class MLPFP8(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.fc = LinearFP8()

# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import torch
from enum import Enum
from src.utils.fp8 import FP8E4M3, FP8E5M2

class DataType(Enum):
	fp32 = torch.float			# E8M23
	fp16 = torch.float16		# E5M10
	bf16 = torch.bfloat16		# E8M7
	fp8 = torch.uint8			# E4M3, E5M2 format, use uint8 to store FP8 bits


class FP8Format(Enum):
	e4m3 = FP8E4M3
	e5m2 = FP8E5M2


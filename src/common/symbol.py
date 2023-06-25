# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import torch
from enum import Enum

class DataType(Enum):
	fp32 = 0		# E8M23
	fp16 = 1		# E5M10
	bf16 = 2		# E8M7
	fp8e4m3 = 3		# E4M3
	fp8e5m2 = 4		# E5M2


data_types = {
	DataType.fp32.name: torch.float,
	DataType.fp16.name: torch.float16,
	DataType.bf16.name: torch.bfloat16,
}

mantissa_bits = {
	DataType.fp32.name: 23,
	DataType.fp16.name: 10,
	DataType.bf16.name: 7,
	DataType.fp8e4m3.name: 3,
	DataType.fp8e5m2.name: 2,
}
# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/27
Description   :
"""
import os, sys
sys.path.append(os.getcwd())

import torch
import torch.cuda as cuda
from src.utils.fp8 import FP8E4M3
from src.utils.bits import get_float_bits

if cuda.is_available():
	device = torch.device(0)
else:
	device = "cpu"

fp32s = [
	0.1,
]

fp32_tensors = [torch.tensor(f, device=device, dtype=torch.float) for f in fp32s]

for fp32 in fp32_tensors:
	fp8 = FP8E4M3.convert_to_fp8(fp32)
	print(f"fp32: [{fp32.item()}] --> fp8: [{fp8.item()}]")
	fp32_bits = get_float_bits(fp32.item(), "fp32")
	fp8_bits = get_float_bits(fp8.item(), "fp8")
	print(f"fp32 bits: [{fp32_bits}] --> fp8: [{fp8_bits}]")
	print()

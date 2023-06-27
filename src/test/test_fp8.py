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
	# # specials
	# float("inf"),
	# float("-inf"),
	# float("nan"),
	# 0,
	# # overflow
	# 448,			# 0 1111 110 = 126
	# 449.1,		# 126
	# 1000.1234,	# 126
	# 10000.324,	# 126
	480,			# 2^8 * 1.875 ==> 126
	# 464,			# 2^8 * 1.8125 ==> 126
	# -12345.3456,	# 254
	# -448,			# 254
	# -448.135,		# 254
	-480,
	# -464,
	# # underflow
	# 0.001953125,	# 2^(-9) ==> 1
	# 0.0009765625, 	# 2^(-10) ==> 0
	# 0.00146484375,	# 1.5*2^(-10) ==> 1
	# 0.0017,			# 1.7408*2^(-10) ==> 1
	# -0.00004,
	# -0.00048828125, # 2^(-11)
	# -0.00146484375,	# 1.5*2^(-10)

	# # normal
	# 0.1,		#
	# 0.0234375,	#
	# -0.002,		# 1.024*2^(-9)
]

# fp32_tensors = [torch.tensor(f, device=device, dtype=torch.float) for f in fp32s]

# for fp32 in fp32_tensors:
# 	fp8 = FP8E4M3.convert_to_fp8(fp32)
# 	print(f"fp32: [{fp32.item()}] --> fp8: [{fp8.item()}]")
# 	fp32_bits = get_float_bits(fp32.item(), "fp32")
# 	fp8_bits = get_float_bits(fp8.item(), "fp8")
# 	print(f"fp32 bits: [{fp32_bits}] --> fp8: [{fp8_bits}]")
# 	print()

fp8s = [
	# # specials
	# 0,		# 0
	# 128,	# 0
	# 127,	# NaN
	# 255,	# NaN
	#
	1,
	10,
	100,
]

fp8_tensors = [torch.tensor(f, device=device, dtype=torch.uint8) for f in fp8s]

for fp8 in fp8_tensors:
	fp32 = FP8E4M3.convert_from_fp8(fp8, fp_type="fp32")
	print(f"fp8: [{fp8.item()}] --> fp32: [{fp32.item()}]")
	fp32_bits = get_float_bits(fp32.item(), "fp32")
	fp8_bits = get_float_bits(fp8.item(), "fp8")
	print(f"fp8 bits: [{fp8_bits}] --> fp32 bits: [{fp32_bits}]")
	print()

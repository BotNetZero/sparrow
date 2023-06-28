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
	# 472,			# 0.1111.110.11  ==> 0.1111.110
	# 480,			# 0.1111.111 ==> 0.1111.110
	# 464,			# 2^8 * 1.8125 ==> 126
	# -12345.3456,	# 254
	# -448,			# 254
	# -448.135,		# 254
	# -480,
	# -464,
	# # underflow
	# 0.0009765625, 	# 2^(-10) ==> 0
	# 0.00146484375,	# 1.5*2^(-10) ==> 1
	# 0.0017,			# 1.7408*2^(-10) ==> 1
	# -0.00004,
	# -0.00048828125, # 2^(-11)
	# -0.00146484375,	# 1.5*2^(-10)

	# normal
	0.017578125,	# 0.0001.001
	-0.018,			# 1.0001.001
	0.01953125,		# 0.0001.010
	-0.0205078125,	# 1.0001.010.1
	0.021484375,	# 0.0001.011
	0.02197265625,	# 1.0001.011.01
	-0.0224609375,	# 1.0001.011.1
	0.0234375,		# 0.0001.100
	0.1,			# 0.0011.101
	0.1015625,		# 0.0011.101
	1.5,			# 0.0111.100
	1.625,			# 0.0111.101
	3,				# 0.1000.100
	# # subnormal
	# 0.001953125,	# 0.0000.001
	# -0.001953125,	# 1.0000.001
	# -0.002,			# 1.0000.001
	# 0.0078125,		# 0.0000.100
	# 0.0087890625,	# 0.0000.100.1 ==> 0.0000.100
	# 0.0029296875,	# 0.0000.001.1 ==> 0.0000.010
	# -0.0068359375,	# 1.0000.011.1 ==> 1.0000.100
	# 0.0146484375, 	# 0.0000.111.1 ==> 0.0001.000
]

fp32_tensors = [torch.tensor(f, device=device, dtype=torch.float) for f in fp32s]

for fp32 in fp32_tensors:
	fp8 = FP8E4M3.convert_to_fp8(fp32)
	print(f"fp32: [{fp32.item()}] --> fp8: [{fp8.item()}]")
	fp32_bits = get_float_bits(fp32.item(), "fp32")
	fp8_bits = get_float_bits(fp8.item(), "fp8")
	print(f"fp32 bits: [{fp32_bits}] --> fp8: [{fp8_bits}]")
	print()


# fp8s = [
# 	# # specials
# 	# 0,		# 0
# 	# 128,	# 0
# 	# 127,	# NaN
# 	# 255,	# NaN
# 	#
# 	# 1,
# 	# 2,
# 	# 3,
# 	# 4,
# 	# 5,
# 	# 6,
# 	# 7,
# 	# 8,
# 	# 9,
# 	# 10,
# 	# 100,
# 	200,
# 	201,
# ]

# fp8_tensors = [torch.tensor(f, device=device, dtype=torch.uint8) for f in fp8s]

# for fp8 in fp8_tensors:
# 	fp32 = FP8E4M3.convert_from_fp8(fp8, fp_type="fp32")
# 	print(f"fp8: [{fp8.item()}] --> fp32: [{fp32.item()}]")
# 	fp32_bits = get_float_bits(fp32.item(), "fp32")
# 	fp8_bits = get_float_bits(fp8.item(), "fp8")
# 	print(f"fp8 bits: [{fp8_bits}] --> fp32 bits: [{fp32_bits}]")
# 	print()

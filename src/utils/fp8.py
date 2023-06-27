# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/26
Description   :
"""
import numpy as np
import torch
from src.common.symbol import DataType, data_types

class FP8E4M3:
	"""
	FP8 E4M3 format:
	normal: sign * 2^(e-7) * 1.f, exp = e-7 = [-6, 8]
	subnormal: sign * 2^(1-7) * 0.f = sign * 2^(-6-m) * 1.f, exp = -6-m
	Using uint8 to store FP8
	positive range: 2^(-9) --> 0.875 * 2^(-6) --> 2^(-6) --> 1.75 * 2^8
							subnormal					normal
	"""
	bias = 7
	# specials
	inf = 126		# clip to fp8_max: 0.1111.110
	inf_mius = 254	# clip to fp8_min: 1.1111.110
	zero = 0
	nan = 127		# 0.1111.111
	#
	exp_max = 8		# s.1111.110 = 1.75*2^8
	exp_min = -6	# s.0001.000 = 2^(-6)
	#
	dtype = data_types[DataType.fp8.name]	# uint8

	@classmethod
	def convert_to_fp8(cls, tensor):
		# specials
		fp8_value = cls.convert_to_fp8_specials(tensor)
		if fp8_value is not None:
			return torch.tensor(fp8_value, dtype=cls.dtype, device=tensor.device)
		#
		if tensor.dtype == torch.float:
			fp8_value = cls.convert_fp32_to_fp8(tensor)
		elif tensor.dtype == torch.float16:
			pass
		elif tensor.dtype == torch.bfloat16:
			pass
		else:
			raise NotImplementedError(f"Not support type conversion: [{tensor.dtype}] --> FP8E4M3")

		return torch.tensor(fp8_value, dtype=cls.dtype, device=tensor.device)

	@classmethod
	def convert_to_fp8_specials(cls, tensor):
		if tensor.item() == float('inf'):
			return cls.inf
		elif tensor.item() == float("-inf"):
			return cls.inf_mius
		elif tensor.item() == float("nan"):
			return cls.nan
		elif tensor.item() == 0:
			return cls.zero
		else:
			return None

	@classmethod
	def convert_fp32_to_fp8(cls, tensor):
		"""
		FP32 E8M23:
		normal: sign * 2^(e-127) * 1.f
		subnormal: sign * 2^(-126) * 0.f = sign * 2^(-126-m) * 1.f
		"""
		src_num = tensor.item()
		num_abs = abs(src_num)
		#
		exp_fp32 = int(np.floor(np.log2(num_abs))) 	# 2^(exp_fp32) = 2^(e-127) or 2^(-126-m)
		if exp_fp32 > cls.exp_max:		# overflow, clip
			e = 15						# s.1111.110
		elif exp_fp32 < cls.exp_min:	#
			e = 0						# s.0000
		else:							# normal range
			e = exp_fp32+cls.bias		# s.0001 ~ s.1111
		#
		exp_fp8 = e - cls.bias
		frac_fp32 = num_abs * (2**-exp_fp8)	# 2^(exp_fp32-exp_fp16) * 1.f
		if frac_fp32 >= 2:					# overflow, clip
			frac = 6						# s.1111.110
		else:
			# round
			shift_num = frac_fp32*(2**3) + 0.5	# left shift 3bits and round
			# tie even
			if int(shift_num) % 2 == 0:
				frac = int(shift_num)
			else:
				frac = int(np.ceil(shift_num))
		#
		fp8_value = e*2**3 + frac
		if src_num < 0:
			fp8_value += 128
		return fp8_value

	@classmethod
	def convert_from_fp8(tensor, fp_type=DataType.fp16.name):
		pass



class FP8E5M2:
	"""
	FP8 E5M2 format: sign * 2^(exp-15) * 1.f
	Using uint8 to store FP8
	"""
	bias = 15
	# specials
	inf = 124		# 0.11111.00
	inf_mius = 252	# 1.11111.00
	zero = 0
	nan = 127		# 0.11111.11
	#
	max = 57344		#
	#
	dtype = data_types[DataType.fp8.name]	# uint8

	@classmethod
	def convert_to_fp8(cls, tensor):
		pass


	@classmethod
	def convert_from_fp8(cls, tensor, fp_type=DataType.fp16.name):
		if tensor.dtype != data_types[DataType.fp8.name]:
			raise TypeError(f"tensor type error: [{tensor.dtype}]")
		pass

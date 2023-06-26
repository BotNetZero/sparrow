# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/26
Description   :
"""
import numpy as np
import torch
from src.common.symbol import DataType, data_types
from src.utils.fp_conversion import get_float_bits

class FP8E4M3:
	"""
	FP8 E4M3 format:
	normal: sign * 2^(e-7) * 1.f, exp = e - 7 = [-6, 8]
	subnormal: sign * 2^(1-6) * 0.f = sign * 2^exp * a.b
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
	frac_max = 7	# s.xxx.111
	frac_min = 0	# s.xxx.000
	#
	dtype = data_types[DataType.fp8e4m3.name]	# uint8

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
		E8M23:
		normal: sign * 2^(exp-127) * 1.f
		subnormal: sign * 2^(-126) * 0.f = sign * 2^(exp) * a.b, where exp < -126
		"""
		src_num = tensor.item()
		num_abs = abs(src_num)
		#
		exp_fp32 = int(np.floor(np.log2(num_abs))) 	# 2^(exp_fp32) = 2^(exp-127) or 2^(exp)
		if exp_fp32 >= cls.exp_max:
			exp = 15			# s.1111
		elif exp_fp32 < cls.exp_min:
			exp = 0				# s.0000
		else:
			exp = exp_fp32 		#
		#
		frac_fp32 = num_abs * (2**-exp)				# 1.f, 0.f


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
	dtype = data_types[DataType.fp8e5m2.name]	# uint8

	@classmethod
	def convert_to_fp8(cls, tensor):
		pass


	@classmethod
	def convert_from_fp8(cls, tensor, fp_type=DataType.fp16.name):
		if tensor.dtype != data_types[DataType.fp8e5m2.name]:
			raise TypeError(f"tensor type error: [{tensor.dtype}]")
		pass

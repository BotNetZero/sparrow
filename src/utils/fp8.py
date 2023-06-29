# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/26
Description   :
"""
import torch
import struct
import numpy as np
from src.common.symbol import DataType


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
	inf_minus = 254	# clip to fp8_min: 1.1111.110
	zero = 0
	nan = 127		# 0.1111.111
	nan_minus = 255	# 1.1111.111
	#
	exp_max = 8		# s.1111.110 = 1.75*2^8
	exp_min = -6	# s.0001.000 = 2^(-6)
	#
	dtype = DataType.fp8.value	# uint8

	@classmethod
	def convert_to_fp8(cls, tensor):
		"""
		convert high precision fp (fp32, fp16, bf16) to fp8
		"""
		if tensor.dtype not in (torch.float, torch.float16, torch.bfloat16):
			raise NotImplementedError(f"Not support type conversion: [{tensor.dtype}] --> FP8E4M3")

		

		# specials
		fp8_value = cls._convert_to_fp8_specials(tensor)
		if fp8_value is not None:
			return torch.tensor(fp8_value, dtype=cls.dtype, device=tensor.device)
		#
		fp8_value = cls._convert_to_fp8(tensor)

		return torch.tensor(fp8_value, dtype=cls.dtype, device=tensor.device)

	@classmethod
	def _convert_to_fp8_specials(cls, tensor):
		if tensor.item() == float('inf'):
			return cls.inf
		elif tensor.item() == float("-inf"):
			return cls.inf_minus
		elif torch.isnan(tensor):	# float("nan"), float("-nan")
			return cls.nan
		elif tensor.item() == 0:
			return cls.zero
		else:
			return None

	@classmethod
	def _convert_to_fp8(cls, tensor):
		"""
		"""
		src_num = tensor.item()
		num_abs = abs(src_num)
		#
		exp_fp32 = int(np.floor(np.log2(num_abs))) 	# e.g.: 2^(exp_fp32) = 2^(e-127) or 2^(-126-m)
		if exp_fp32 > cls.exp_max:		# overflow, clip
			e = 15						# s.1111.110
			exp_fp8 = 8					# 15-cls.bias
		elif exp_fp32 < cls.exp_min:	# subnormal + underflow
			e = 0						# s.0000
			exp_fp8 = -6				# 1-cls.bias
		else:							# normal range
			e = exp_fp32+cls.bias		# s.0001 ~ s.1111
			exp_fp8 = exp_fp32			#
		#
		frac_fp8 = num_abs * (2**-exp_fp8)		# 2^(exp_fp32-exp_fp8) * 1.f
		if frac_fp8 >= 2.0:						# exp_fp32-exp_fp8 >= 1, overflow, clip
			frac = 6							# s.1111.110
		else:									# [0,2)
			if frac_fp8 >= 1.0:					# 1.f
				frac_fp8 -= 1.0					# 0.f
			# RNE
			shift_num = frac_fp8*(2**3)			# left shift 3 bits
			floor_num = np.floor(shift_num)
			int_num = int(floor_num)
			delta = shift_num - floor_num
			if delta > 0.5:						# > midpoint
				frac = int_num + 1
			elif delta == 0.5:					# midpoint, tie even
				if int_num % 2 == 0:			# int operation
					frac = int_num
				else:
					frac = int_num + 1
			else:								# < midpoint
				frac = int_num
			# clip: when e=15(1111), frac<7(111)
			if e == 15:
				frac = min(frac, 6)
		#
		fp8_value = e*(2**3) + frac
		if src_num < 0:
			fp8_value += 128
		return fp8_value

	@classmethod
	def convert_from_fp8(cls, tensor, fp_type=DataType.fp16.name):
		"""
		convert fp8 to fp32, fp16, bf16
		"""
		fp_type = fp_type.lower()
		if fp_type not in (DataType.fp32.name, DataType.fp16.name, DataType.bf16.name):
			raise NotImplementedError(f"not support type conversion: FP8E4M3 --> [{fp_type}]")
		dtype = DataType[fp_type].value
		#
		fp_value = cls._convert_from_fp8_specials(tensor)
		if fp_value is not None:
			return torch.tensor(fp_value, dtype=dtype, device=tensor.device)

		fp_value = cls._convert_from_fp8(tensor)

		return torch.tensor(fp_value, dtype=dtype, device=tensor.device)

	@classmethod
	def _convert_from_fp8_specials(cls, tensor):
		"""
		"""
		# no +/- Inf in E4M3 format, all clipped to normal max
		if tensor.item() == cls.nan:
			return float("nan")
		if tensor.item() == cls.nan_minus:
			return float("-nan")
		if tensor.item() == 0:
			return 0

		return None

	@classmethod
	def _convert_from_fp8(cls, tensor):
		"""
		translate uint8 to real number FP8
		No any operations for conversing low FP to high FP
		"""
		if tensor.dtype != cls.dtype:
			raise TypeError(f"input tensor type [{tensor.dtype}] error, it is not FP8!!!")
		#
		binary_fp8 = struct.pack('!B', tensor.item())			# uint8
		sign = (int.from_bytes(binary_fp8, 'big') >> 7) & 1
		s = 1 if sign == 0 else -1
		#
		exp  = (int.from_bytes(binary_fp8, 'big') >> 3) & 0xF	# E4
		mant = int.from_bytes(binary_fp8, 'big') & 0x7			# M3
		#
		frac = mant * (2**(-3))		# right shift 3 bits
		if exp > 0:					# normal
			return s * 2**(exp-cls.bias) * (1+frac)
		else:						# subnormal
			return s * 2**(-6) * frac


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
	dtype = DataType.fp8.value	# uint8

	@classmethod
	def convert_to_fp8(cls, tensor):
		pass


	@classmethod
	def convert_from_fp8(cls, tensor, fp_type=DataType.fp16.name):
		if tensor.dtype != cls.dtype:
			raise TypeError(f"tensor type error: [{tensor.dtype}]")
		pass

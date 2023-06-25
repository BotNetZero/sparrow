# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import struct
import numpy as np
import torch
from src.common.symbol import DataType, data_types


def convert_to_fp8(tensor, fp8_type="E4M3"):
	"""
	convert real number to fp8
	based on IEEE 754, real = sign * 2^exp * 1.f

	fp32: E8M23
	fp16: E5M10
	bf16: E8M7
	fp8: E4M3, E5M2
	"""
	if tensor.dtype == DataType.fp32.name:
		pass


def _convert_fp32_to_fp8(tensor, fp8_type):
	"""
	fp32: E8M23
	"""
	pass

def _convert_fp16_to_fp8(tensor, fp8_type):
	"""
	fp16: E5M10
	"""
	pass

def _convert_bf16_to_fp8(tensor, fp8_type):
	"""
	bf16: E8M7
	"""
	pass



def convert_from_fp8(tensor, fp_type=DataType.fp16.name):
	"""
	convert fp8 to higher precision fp (fp32, fp16, bf16)
	"""
	pass


def _convert_fp8_to_fp32(tensor):
	pass

def _convert_fp8_to_fp16(tensor):
	pass

def _convert_fp8_to_bf16(tensor):
	pass


def convert_fp(tensor, fp_dtype):
	"""
	convert real number to one fp dtype
	based on IEEE 754, real = sign * 2^exp * 1.f

	fp32: E8M23
	fp16: E5M10
	bf16: E8M7
	fp8: E4M3, E5M2
	"""
	if fp_dtype == DataType.fp32.name:
		shift_bits = 23
	elif fp_dtype == DataType.fp16.name:
		shift_bits = 10
	elif fp_dtype == DataType.bf16.name:
		shift_bits = 7
	elif fp_dtype == DataType.fp8e4m3.name:
		shift_bits = 3
	elif fp_dtype == DataType.fp8e5m2.name:
		shift_bits = 2
	else:
		raise NotImplementedError(f"not support dtype [{fp_dtype}]")
	#
	src_num = tensor.item()
	src_dtype = tensor.dtype
	if src_dtype == data_types[fp_dtype]:	#
		return tensor
	#
	sign = 1 if src_num >= 0 else -1
	num_abs = abs(src_num)
	exp = int(np.floor(np.log2(num_abs)))	# 2^exp
	frac = num_abs * 2**-exp - 1			# 0.f
	#

	#
	frac_shift = int(frac*(2**shift_bits))		# left shift
	# frac_bits = bin(frac_shift)

	new_frac = 1 + frac_shift/(2**shift_bits)	# right shift
	c = torch.tensor(sign*(2**exp)*new_frac, device=tensor.device, dtype=data_types[fp_dtype])	#
	return c


def get_float_bits(float_number, dtype):
	"""
	get bits representation of fp number
	"""
	dtype = dtype.lower()
	# fp32
	if dtype == DataType.fp32.name:
		float_bits = _get_fp32_bits(float_number)
	# fp16
	elif dtype == DataType.fp16.name:
		float_bits = _get_fp16_bits(float_number)
	# bf16
	elif dtype == DataType.bf16.name:
		float_bits = _get_bf16_bits(float_number)
	# fp8
	elif dtype == DataType.fp8.name:
		pass
	else:
		raise NotImplementedError(f"not support dtype [{dtype}]")

	return float_bits

def _get_fp32_bits(float_number):
	"""
	"""
	# Convert the floating-point number to its binary representation
	float_bytes = struct.pack("!f", float_number)

	# Convert the binary representation to a string of bits
	float_bits = ''.join(format(byte, '08b') for byte in float_bytes)

	return float_bits

def _get_fp16_bits(float_number):
	"""
	"""
	# Convert the floating-point number to its binary representation
	float_bytes = struct.pack("!e", float_number)

	# Convert the binary representation to a string of bits
	float_bits = ''.join(format(byte, '08b') for byte in float_bytes)

	return float_bits


def _get_bf16_bits(float_number):
	"""
	"""
	# 32bit
	float_bytes = struct.pack("!f", float_number)
	# truncate
	float_bits = ''.join(format(byte, '08b') for byte in float_bytes[0:2])
	return float_bits


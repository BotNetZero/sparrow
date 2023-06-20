# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import struct
import numpy as np
from src.common.symbol import DataType


# def convert_fp(real, fp_dtype):
# 	"""
# 	based on IEEE 754, real = sign * 2^exp * 1.f
# 	"""
# 	sign = 1 if real >= 0 else -1
# 	fp16_abs = abs(real)
# 	exponent = int(np.floor(np.log2(fp16_abs))) - 7
# 	mantissa = int(round(fp16_abs * 2 ** -exponent * 128)) & 0xFF
# 	fp8_value = sign * (exponent + 128) + mantissa

def convert_fp(num, fp_type):
	"""
	convert real number to one fp dtype
	based on IEEE 754, real = sign * 2^exp * 1.f
	"""
	sign = 1 if num >= 0 else -1
	num_abs = abs(num)
	exp = int(np.floor(np.log2(num_abs)))
	fract = num_abs * 2**-exp
	#
	if fp_type == DataType.fp32:
		pass


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


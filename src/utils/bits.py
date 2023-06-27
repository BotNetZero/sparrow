# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import struct
from src.common.symbol import DataType


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
		float_bits = _get_fp8_bits(float_number)
	else:
		raise NotImplementedError(f"not support dtype [{dtype}]")

	return float_bits

def _get_fp32_bits(float_number):
	"""
	"""
	# Convert the floating-point number to its binary representation
	float_hex = struct.pack("!f", float_number)

	# Convert the binary representation to a string of bits
	float_bits = ''.join(format(byte, '08b') for byte in float_hex)

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


def _get_fp8_bits(number):
	"""
	Because fp8 is stored in uint8, number is not float, is unsigned int
	"""
	if not is_integer(number):
		raise ValueError(f"converted number [{number}] is not integer")
	#
	uint_bytes = struct.pack("!B", number)
	bits = ''.join(format(byte, '08b') for byte in uint_bytes)
	return bits


def is_integer(num):
	try:
		float_num = float(num)
		return float_num.is_integer()
	except ValueError:
		return False

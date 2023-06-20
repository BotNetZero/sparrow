# -*- coding: utf-8 -*-
"""
Author        : Di Niu
CreatedDate   : 2023/06/20
Description   :
"""
import torch
from enum import Enum

class DataType(Enum):
    fp32 = 0
    fp16 = 1
    bf16 = 2
    fp8 = 3
    int32 = 4
    int8 = 5

data_types = {
    DataType.fp32.name: torch.float,
    DataType.fp16.name: torch.float16,
    DataType.bf16.name: torch.bfloat16,
}


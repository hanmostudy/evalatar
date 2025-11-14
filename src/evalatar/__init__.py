"""
Evalatar: 一个用于评估数字人视频生成模型性能的Python库。
"""

__version__ = "0.1.0"

from .fid import calculate_fid
from .fvd import calculate_fvd
from .sync import calculate_sync_c, calculate_sync_d
from .iqa import calculate_iqa
from .ase import calculate_ase
from .csim import calculate_csim

__all__ = [
    'calculate_fid',
    'calculate_fvd',
    'calculate_sync_c',
    'calculate_sync_d',
    'calculate_iqa',
    'calculate_ase',
    'calculate_csim',
    ]
'''
MRI 序列编号规则
序列识别逻辑
'''

from pathlib import Path
from configs.global_config import ALL_SEQUENCES

def identify_sequence(nii_path: Path):
    """
    根据文件名识别 MRI 序列
    返回：序列 index
    """
    name = nii_path.name.upper()

    for idx, seq_name in enumerate(ALL_SEQUENCES):
        if seq_name.upper() in name:
            return idx + 1  # 序列编号从 1 开始

    return None

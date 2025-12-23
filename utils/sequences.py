'''
MRI 序列编号规则
序列识别逻辑
'''

from pathlib import Path

# ===== 序列名称到编号的映射 =====
SEQUENCE_MAP = {
    "T1WI": 1,
    "T1": 1,
    "T2WI": 2,
    "T2": 2,
    "FLAIR": 3,
    "DWI": 4,
    "+C": 5,
}

def identify_sequence(nii_path: Path):
    """根据文件名判断 MRI 序列类型"""
    name = nii_path.name.upper()
    for key, seq_id in SEQUENCE_MAP.items():
        if key in name:
            return seq_id
    return None
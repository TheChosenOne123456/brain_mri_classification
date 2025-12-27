'''
扫描原始目录
收集 case 文件夹
'''

def collect_cases(src_dirs):
    """遍历源目录，收集所有 case 文件夹"""
    cases = []
    for src_dir in src_dirs:
        if not src_dir.exists():
            continue
        for case_dir in src_dir.iterdir():
            if case_dir.is_dir():
                cases.append(case_dir)
    return sorted(cases)
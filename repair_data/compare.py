import pandas as pd
import os

# ================= 路径配置 =================
BASE_DIR = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data"
ORI_PATH = os.path.join(BASE_DIR, "ori", "test.csv")
# REPAIR_PATH = os.path.join(BASE_DIR, "repair", "test.csv")
REPAIR_PATH = os.path.join(BASE_DIR, "repair_nli_v2", "test.csv")
# 输出差异文件的路径
# DIFF_OUTPUT_PATH = os.path.join(BASE_DIR, "repair", "test_changes.csv")
# DIFF_OUTPUT_PATH = os.path.join(BASE_DIR, "repair_nli", "test_changes.csv")
DIFF_OUTPUT_PATH = os.path.join(BASE_DIR, "repair_nli_v2", "test_changes.csv")

def extract_differences():
    print(f"正在读取原始文件: {ORI_PATH}")
    if not os.path.exists(ORI_PATH) or not os.path.exists(REPAIR_PATH):
        print("错误：找不到原始文件或修复后的文件，请检查路径。")
        return

    # 读取文件
    df_ori = pd.read_csv(ORI_PATH)
    df_repair = pd.read_csv(REPAIR_PATH)

    # 预处理：为了确保行对齐，必须对原始数据做与之前完全一致的 dropna 操作
    # 之前的脚本逻辑是: df.dropna(subset=['text', 'label'])
    # 只有这样，两个 dataframe 的行数和顺序才能一一对应
    df_ori_clean = df_ori.dropna(subset=['text', 'label']).reset_index(drop=True)
    df_repair_clean = df_repair.dropna(subset=['text', 'label']).reset_index(drop=True)

    # 检查行数是否一致
    if len(df_ori_clean) != len(df_repair_clean):
        print(f"警告：行数不匹配！原始清洗后 {len(df_ori_clean)} 行，修复后 {len(df_repair_clean)} 行。")
        print("尝试通过 'index' 列（如果存在）进行对齐...")
        # 如果存在 index 列，建议 merge，这里简单处理，若不匹配可能无法直接对比
    
    print("正在对比标签差异...")
    
    # 核心逻辑：找出 label 不相等的行
    # 注意：确保数据类型一致（都转为int）
    diff_mask = df_ori_clean['label'].astype(int) != df_repair_clean['label'].astype(int)
    
    # 提取差异行
    df_diff = df_ori_clean[diff_mask].copy()
    
    # 将修改后的新标签加进去，方便对比
    df_diff['new_label'] = df_repair_clean.loc[diff_mask, 'label'].values
    
    # 重命名原标签列，清晰一点
    df_diff.rename(columns={'label': 'original_label'}, inplace=True)

    # 整理列顺序，把 original_label 和 new_label 放在最前面方便查看
    cols = list(df_diff.columns)
    # 移除要置顶的列
    for c in ['original_label', 'new_label', 'text']:
        if c in cols: cols.remove(c)
    # 重新组合：ID类列 + 原标签 + 新标签 + 文本 + 其他
    final_cols = ['original_label', 'new_label', 'text'] + cols
    # 如果原数据有 'index' 或 'id' 列，也可以把它放在最前面
    if 'index' in df_diff.columns:
        final_cols.insert(0, 'index')
        
    df_diff = df_diff[final_cols]

    # 输出统计信息
    print(f"检测到 {len(df_diff)} 处标签修改。")
    
    # 保存
    print(f"正在保存差异对比文件到: {DIFF_OUTPUT_PATH}")
    df_diff.to_csv(DIFF_OUTPUT_PATH, index=False)
    print("完成！")

if __name__ == "__main__":
    extract_differences()
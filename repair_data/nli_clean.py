import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os

# ================= 核心配置 =================
# 显存充足（>12G）推荐用 batch_size=32 或 64
# 如果爆显存 (OOM)，请改成 8 或 16
BATCH_SIZE = 32 

# 模型：专门用于逻辑推理的强模型
MODEL_NAME = "facebook/bart-large-mnli"

# 阈值：只有模型确信度 > 90% 时才修改，防止乱改
CONFIDENCE_THRESHOLD = 0.90

# 路径配置 (根据你提供的路径)
BASE_DIR = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data"
INPUT_DIR = os.path.join(BASE_DIR, "ori")
OUTPUT_DIR = os.path.join(BASE_DIR, "repair_nli")

# 标签定义
CANDIDATE_LABELS = ["negative review", "positive review"]
LABEL_MAP = {"negative review": 0, "positive review": 1}

# ===========================================

def clean_file(filename, classifier):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f"跳过: 文件不存在 {input_path}")
        return

    print(f"\n正在读取: {filename} ...")
    df = pd.read_csv(input_path)
    
    # 预处理：删掉空行
    df = df.dropna(subset=['text', 'label'])
    
    # 准备数据
    texts = df['text'].astype(str).tolist()
    original_labels = df['label'].astype(int).tolist()
    
    print(f"  正在使用 {MODEL_NAME} 推理 {len(texts)} 条数据 (Batch Size: {BATCH_SIZE})...")
    
    # 使用 pipeline 批量推理
    # 这一步会比较耗时，取决于显卡性能
    results = []
    # 使用 tqdm 显示进度条
    for out in tqdm(classifier(texts, CANDIDATE_LABELS, batch_size=BATCH_SIZE), total=len(texts)):
        results.append(out)

    print("  推理完成，正在修正标签...")
    
    final_labels = []
    fix_count = 0
    
    # 遍历结果进行修正
    for idx, res in enumerate(results):
        # res['labels'][0] 是得分最高的类别
        pred_category = res['labels'][0]
        pred_score = res['scores'][0]
        pred_label = LABEL_MAP[pred_category]
        
        ori_label = original_labels[idx]
        
        # 修正逻辑：预测与原标不同，且置信度极高
        if pred_label != ori_label and pred_score > CONFIDENCE_THRESHOLD:
            final_labels.append(pred_label)
            fix_count += 1
        else:
            final_labels.append(ori_label)

    # 保存结果
    df['label'] = final_labels
    
    print(f"  [{filename}] 修正统计: {fix_count}/{len(df)} (占比 {fix_count/len(df):.2%})")
    print(f"  写入文件 -> {output_path}")
    df.to_csv(output_path, index=False)


def main():
    # 检查 GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"当前运行设备: {'GPU' if device == 0 else 'CPU'}")
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    # 加载模型
    print(f"正在加载 NLI 模型 (首次运行会自动下载)...")
    classifier = pipeline("zero-shot-classification", 
                          model=MODEL_NAME, 
                          device=device)

    # 依次处理三个文件
    for f in ["train.csv", "valid.csv", "test.csv"]:
        clean_file(f, classifier)

    print("\n全部任务完成！")

if __name__ == "__main__":
    main()
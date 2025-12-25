import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import os
import re

# ================= 核心配置 =================
# 显存充足（>12G）推荐用 batch_size=32 或 64
# 如果爆显存 (OOM)，请改成 8 或 16
BATCH_SIZE = 32 

# 模型：专门用于逻辑推理的强模型
MODEL_NAME = "facebook/bart-large-mnli"

# 阈值：只有模型确信度 > 90% 时才考虑修改
CONFIDENCE_THRESHOLD = 0.90

# 路径配置 (根据你提供的路径)
BASE_DIR = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data"
INPUT_DIR = os.path.join(BASE_DIR, "ori")
OUTPUT_DIR = os.path.join(BASE_DIR, "repair_nli_v2") # 建议换个输出目录名区分版本

# 标签定义
CANDIDATE_LABELS = ["negative review", "positive review"]
LABEL_MAP = {"negative review": 0, "positive review": 1}

# ================= 风险过滤器 V2.0 (High-Level Semantic Filter) =================
def get_risk_status(text):
    """
    判断文本是否属于“高风险”类别（高阶语义、反讽、文学评论等）。
    返回: (True/False, reason)
    True 表示有风险，建议跳过修改；False 表示安全，可以修改。
    """
    text = str(text)
    text_lower = text.lower()
    
    # 1. 激动/大写过滤器 (Excessive Caps)
    # 逻辑：长度>20，大写占比>50%，且包含感叹号 -> 视为激动而非愤怒
    upper_count = sum(1 for c in text if c.isupper())
    if len(text) > 20 and (upper_count / len(text) > 0.5) and "!" in text:
         return True, "Excessive Caps/Excitement"
    
    # 特殊俚语保护
    if "cut it up def" in text_lower: 
        return True, "Slang/Jargon"

    # 2. 体验/尺寸过滤器 (Mixed Experience)
    # 逻辑：既有 "love/nice" 又有 "size/fit" -> 属于尺码不合但产品好的模糊地带
    if ("size" in text_lower or "fit" in text_lower) and \
       ("love" in text_lower or "nice" in text_lower or "great" in text_lower):
        return True, "Mixed Experience (Size/Service)"

    # 3. 文学/剧情深度过滤器 (Literary/Plot Context)
    # A. 语境词：暗示这是在讲书/故事
    context_keywords = ['book', 'story', 'novel', 'read', 'chapter', 'author', 'character', 'plot', 'study of', 'film', 'movie']
    # B. 内容词：容易被误判的负面词
    content_keywords = [
        'horror', 'devastation', 'guilt', 'death', 'grim', 'tragedy', 
        'cruel', 'poverty', 'banned', 'adultery', 'slams', 'depressing', 'violent'
    ]
    
    has_context = any(word in text_lower for word in context_keywords)
    has_content = any(word in text_lower for word in content_keywords)
    
    # 特别针对人名/书名 (如 Dickens, Scarlet Letter)
    special_context = any(name in text_lower for name in ['dickens', 'scarlet letter', 'shakespeare', 'twain'])

    # 逻辑：(有语境词 OR 有特殊书名) AND 有负面内容词 -> 判定为剧情描述
    if (has_context or special_context) and has_content:
        #例外：除非明确说了 "boring" 或 "waste"，否则视为剧情描述，不能改
        if "boring" not in text_lower and "waste of" not in text_lower:
            return True, "Literary/Plot Context"

    return False, "Safe"

# ===========================================================================

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
    
    results = []
    # 使用 pipeline 批量推理
    for out in tqdm(classifier(texts, CANDIDATE_LABELS, batch_size=BATCH_SIZE), total=len(texts)):
        results.append(out)

    print("  推理完成，正在执行[修正 + 风险过滤]逻辑...")
    
    final_labels = []
    fix_count = 0        # 实际修改的数量
    skipped_risky_count = 0 # 拦截的高风险数量
    
    # 遍历结果进行修正
    for idx, res in enumerate(results):
        # 1. 获取模型预测
        pred_category = res['labels'][0]
        pred_score = res['scores'][0]
        pred_label = LABEL_MAP[pred_category]
        ori_label = original_labels[idx]
        current_text = texts[idx]
        
        # 2. 判断是否需要修改
        # 条件 A: 模型预测与原标签不同
        # 条件 B: 模型置信度极高 (>0.9)
        if pred_label != ori_label and pred_score > CONFIDENCE_THRESHOLD:
            
            # 3. [新增] 风险过滤器检查
            is_risky, risk_reason = get_risk_status(current_text)
            
            if is_risky:
                # 触发了风险过滤，强制保留原标签（不做修改）
                final_labels.append(ori_label)
                skipped_risky_count += 1
                # (可选) 如果想看拦截了什么，取消下面这行的注释
                # print(f"-> [拦截] index:{idx} 原因:{risk_reason} | 文本:{current_text[:30]}...")
            else:
                # 安全，执行修改
                final_labels.append(pred_label)
                fix_count += 1
        else:
            final_labels.append(ori_label)

    # 保存结果
    df['label'] = final_labels
    
    # 打印详细统计
    total = len(df)
    print(f"-"*40)
    print(f"  [{filename}] 处理报告:")
    print(f"  > 总数据量: {total}")
    print(f"  > NLI模型建议修改数: {fix_count + skipped_risky_count}")
    print(f"  > 实际执行修正数 (Safe): {fix_count} (占比 {fix_count/total:.2%})")
    print(f"  > 风险拦截数 (Skipped): {skipped_risky_count} (避免了潜在的误改)")
    print(f"-"*40)
    
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
    print(f"正在加载 NLI 模型 (facebook/bart-large-mnli)...")
    classifier = pipeline("zero-shot-classification", 
                          model=MODEL_NAME, 
                          device=device)

    # 依次处理三个文件
    for f in ["train.csv", "valid.csv", "test.csv"]:
        clean_file(f, classifier)

    print("\n全部任务完成！")

if __name__ == "__main__":
    main()
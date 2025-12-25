import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# 尝试导入 cleanlab，如果没有则回退到阈值法
try:
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
    print("成功加载 Cleanlab 库，将进行高精度去噪。")
except ImportError:
    print("未检测到 cleanlab 库，将使用置信度阈值进行修正。建议 pip install cleanlab")
    CLEANLAB_AVAILABLE = False

# ================= 路径配置 =================
# 输入目录 (原始数据)
INPUT_DIR = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data/ori"
# 输出目录 (修正后数据)
OUTPUT_DIR = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data/repair"

# 自动创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: {OUTPUT_DIR}")

# 修正阈值：模型置信度超过 90% 才修改标签，防止改错
CONFIDENCE_THRESHOLD = 0.90

def process_and_repair(filename, is_train=False, vectorizer=None, model=None):
    """
    读取原始文件 -> 识别错误 -> 保存到新路径
    """
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    print(f"\n正在处理文件: {filename} ...")
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return None, None

    df = pd.read_csv(input_path)
    
    # 简单预处理：去除完全为空的行
    df = df.dropna(subset=['text', 'label'])
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).to_numpy()
    
    # 1. 获取预测概率
    if is_train:
        print("  [Train] 正在进行交叉验证提取特征 (这可能需要几分钟)...")
        # 训练集：使用 5折交叉验证获取“干净”的预测概率
        vec = TfidfVectorizer(max_features=20000, stop_words='english')
        X = vec.fit_transform(texts)
        # 注意：sklearn 的 LogisticRegression 通常支持 n_jobs=-1，如果这里也报错，请改为 n_jobs=1
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        
        # 获取验证集概率
        pred_probs = cross_val_predict(clf, X, labels, cv=5, method='predict_proba', n_jobs=-1)
        
        # 重新在全量数据上训练，供 valid/test 使用
        clf.fit(X, labels)
        vectorizer = vec
        model = clf
    else:
        # 验证/测试集：使用训练集训练好的模型
        print(f"  [{filename}] 使用训练集模型进行推理...")
        X = vectorizer.transform(texts)
        pred_probs = model.predict_proba(X)

    # 2. 识别并修正错误
    corrected_count = 0
    new_labels = labels.copy()

    if CLEANLAB_AVAILABLE:
        # 使用 cleanlab 算法寻找标签噪声
        print("  使用 Cleanlab 算法分析标签噪声...")
        issues_mask = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by=None,
            n_jobs=1  # <--- 修改处：强制单核运行，避免 assert 错误
        )
        
        for idx, is_issue in enumerate(issues_mask):
            if is_issue:
                prediction = np.argmax(pred_probs[idx])
                confidence = pred_probs[idx][prediction]
                
                # 双重保险：Cleanlab 认为有问题 + 模型置信度 > 90%
                if prediction != labels[idx] and confidence > CONFIDENCE_THRESHOLD:
                    new_labels[idx] = prediction
                    corrected_count += 1
    else:
        # 备选方案：直接使用置信度阈值
        print("  使用置信度阈值分析...")
        predictions = np.argmax(pred_probs, axis=1)
        max_probs = np.max(pred_probs, axis=1)
        
        for idx in range(len(labels)):
            if predictions[idx] != labels[idx] and max_probs[idx] > CONFIDENCE_THRESHOLD:
                new_labels[idx] = predictions[idx]
                corrected_count += 1

    # 3. 应用修正并保存
    df['label'] = new_labels
    
    print(f"  完成! 共修正 {corrected_count} 条数据 (占比 {corrected_count/len(df):.2%})")
    print(f"  正在保存到 -> {output_path}")
    df.to_csv(output_path, index=False)
    
    return vectorizer, model

# ================= 执行流程 =================

# 1. 处理 Train (作为基准模型)
vec, clf = process_and_repair("train.csv", is_train=True)

# 2. 处理 Valid (使用 Train 的逻辑)
if vec and clf:
    process_and_repair("valid.csv", is_train=False, vectorizer=vec, model=clf)

# 3. 处理 Test
if vec and clf:
    process_and_repair("test.csv", is_train=False, vectorizer=vec, model=clf)

print("\n所有操作完成！请检查 repair 文件夹。")
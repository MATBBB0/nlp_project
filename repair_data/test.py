import pandas as pd
from transformers import AutoTokenizer

# 1. 设置
model_checkpoint = "distilbert-base-uncased"
csv_file_path = "/HOME/sysu_grwang/sysu_grwang_1/HDD_POOL/zjq/nlp-project/data/ori/train.csv" # 确保文件名是对的
check_count = 10000 # 只检查前100条

# 2. 加载 Tokenizer
print("加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 3. 读取数据 (只读前 N 条)
try:
    # nrows 参数可以直接让 pandas 只读前几行，速度极快
    df = pd.read_csv(csv_file_path, nrows=check_count)
    print(f"已读取前 {len(df)} 条数据进行抽查。")
except Exception as e:
    print(f"读取文件出错: {e}")
    exit()

# 4. 快速循环检查
over_limit_count = 0

print("-" * 50)
print(f"正在检查前 {len(df)} 条数据中是否有长度超过 512 的...")

for index, row in df.iterrows():
    text = str(row['text'])
    
    # 关键：truncation=False 才能看到真实长度
    tokens = tokenizer(text, truncation=False, padding=False)
    length = len(tokens['input_ids'])
    
    if length > 512:
        over_limit_count += 1
        print(f"\n[发现超长数据] Index: {row.get('index', index)}") # 如果csv里有index列就用，没有就用行号
        print(f"-> 真实长度: {length} tokens")
        print(f"-> 溢出部分: {length - 512} tokens")
        # 打印一下被截断的最后一段话是什么，看看重要不重要
        print(f"-> 结尾内容预览: ...{text[-100:]}") 

print("-" * 50)
if over_limit_count == 0:
    print(f"在前 {len(df)} 条数据中，没有发现超过 512 tokens 的文本。")
    print("提示：这不代表整个数据集没有，但至少说明超长文本不是很密集。")
else:
    print(f"在前 {len(df)} 条数据中，共发现 {over_limit_count} 条超长数据。")
    print("建议：如果溢出部分包含总结性语句（如“don't buy”），请使用 Head+Tail 截断策略。")
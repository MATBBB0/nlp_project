import os
import re
import bz2
import csv
import random
import argparse


def clean_text(text: str) -> str:
    """Strip HTML tags, URLs, and special symbols except .,?!"""
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s\.\,\?\!]", " ", text)  # drop other symbols
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_split(input_bz2_path, output_csv_path, add_header=True):
    """
    将一个 .txt.bz2 文件转换为 CSV，列为 [text, label]。
    不做划分，仅简单写出全部样本。
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with bz2.open(input_bz2_path, "rt", encoding="utf-8", errors="ignore") as fin, \
         open(output_csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if add_header:
            writer.writerow(["text", "label"])

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # 期望格式: "__label__2  some text ..."
            try:
                label_token, text = line.split(" ", 1)
            except ValueError:
                # 没有空格的异常行直接跳过
                continue

            if label_token == "__label__1":
                label = 0  # negative
            elif label_token == "__label__2":
                label = 1  # positive
            else:
                # 遇到未知 label 直接跳过
                continue

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            writer.writerow([cleaned_text, label])


def split_train_valid(
    input_bz2_path,
    train_csv_path,
    valid_csv_path,
    valid_ratio=0.1,
    seed=42,
):
    """
    从 train.ft.txt.bz2 中流式划分 train/valid：
    - 约 valid_ratio 比例写入 valid.csv
    - 剩下写入 train.csv
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)

    with bz2.open(input_bz2_path, "rt", encoding="utf-8", errors="ignore") as fin, \
         open(train_csv_path, "w", newline="", encoding="utf-8") as ftrain, \
         open(valid_csv_path, "w", newline="", encoding="utf-8") as fvalid:
        train_writer = csv.writer(ftrain)
        valid_writer = csv.writer(fvalid)

        train_writer.writerow(["text", "label"])
        valid_writer.writerow(["text", "label"])

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                label_token, text = line.split(" ", 1)
            except ValueError:
                continue

            if label_token == "__label__1":
                label = 0
            elif label_token == "__label__2":
                label = 1
            else:
                continue

            cleaned_text = clean_text(text)
            if not cleaned_text:
                continue

            # 按概率划到 valid/train，避免一次性读入内存
            if random.random() < valid_ratio:
                valid_writer.writerow([cleaned_text, label])
            else:
                train_writer.writerow([cleaned_text, label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="包含 train.ft.txt.bz2 / test.ft.txt.bz2 的目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 CSV 的目录",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.05,
        help="从训练集中划分为验证集的比例（默认 0.05）",
    )
    args = parser.parse_args()

    train_bz2 = os.path.join(args.data_dir, "train.ft.txt.bz2")
    test_bz2 = os.path.join(args.data_dir, "test.ft.txt.bz2")

    train_csv = os.path.join(args.output_dir, "train.csv")
    valid_csv = os.path.join(args.output_dir, "valid.csv")
    test_csv = os.path.join(args.output_dir, "test.csv")

    print(f"Splitting train/valid from: {train_bz2}")
    split_train_valid(train_bz2, train_csv, valid_csv, valid_ratio=args.valid_ratio)

    print(f"Converting test split from: {test_bz2}")
    convert_split(test_bz2, test_csv, add_header=True)

    print("Done!")
    print("Train CSV:", train_csv)
    print("Valid CSV:", valid_csv)
    print("Test  CSV:", test_csv)


if __name__ == "__main__":
    main()

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Example:
    index: int
    text: str
    label: int


def load_test_examples(csv_path: Path, text_column: str, label_column: str, encoding: str) -> List[Example]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")

    examples: List[Example] = []
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                label_value = int(row[label_column])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid or missing label in row {idx} of {csv_path}") from exc

            examples.append(Example(index=idx, text=row[text_column], label=label_value))
    return examples


def load_predictions(prediction_path: Path, encoding: str) -> Dict[int, int]:
    predictions: Dict[int, int] = {}

    with prediction_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "index" not in reader.fieldnames or "prediction" not in reader.fieldnames:
            raise ValueError(f"Prediction file {prediction_path} must contain 'index' and 'prediction' columns.")

        for row in reader:
            idx = int(row["index"])
            predictions[idx] = int(row["prediction"])
    return predictions


def write_wrong_cases(wrong_cases: List[Example], predictions: Dict[int, int], output_path: Path, encoding: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=encoding, newline="") as f:
        fieldnames = ["index", "label", "prediction", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in wrong_cases:
            writer.writerow({
                "index": case.index,
                "label": case.label,
                "prediction": predictions[case.index],
                "text": case.text,
            })


def collect_wrong_cases(
    output_root: Path,
    test_examples: List[Example],
    prediction_filename: str,
    mistake_filename: str,
    encoding: str,
) -> None:
    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    for model_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        prediction_path = model_dir / prediction_filename
        if not prediction_path.exists():
            continue

        predictions = load_predictions(prediction_path, encoding=encoding)
        wrong_cases = [
            example for example in test_examples
            if predictions.get(example.index) is not None and predictions[example.index] != example.label
        ]

        mistake_path = model_dir / mistake_filename
        write_wrong_cases(wrong_cases, predictions, mistake_path, encoding=encoding)
        print(f"{model_dir.name}: {len(wrong_cases)} wrong samples written to {mistake_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect misclassified samples for every model output directory."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("output"),
        help="Root directory that contains per-model output folders.",
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("data/test.csv"),
        help="CSV file that holds gold labels (expects 'text' and 'label' columns).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column inside the test CSV.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column inside the test CSV.",
    )
    parser.add_argument(
        "--prediction_filename",
        type=str,
        default="predict_results.txt",
        help="Filename of the prediction file inside each model directory.",
    )
    parser.add_argument(
        "--mistake_filename",
        type=str,
        default="wrong_predictions.csv",
        help="Filename for the generated misclassification file.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding used for reading/writing CSV/TSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_csv = args.test_csv
    if not test_csv.exists():
        alt_path = Path("data/ori/test.csv")
        if alt_path.exists():
            test_csv = alt_path
        else:
            raise FileNotFoundError(f"Cannot find test CSV at {args.test_csv} or {alt_path}")

    examples = load_test_examples(
        csv_path=test_csv,
        text_column=args.text_column,
        label_column=args.label_column,
        encoding=args.encoding,
    )

    collect_wrong_cases(
        output_root=args.output_root,
        test_examples=examples,
        prediction_filename=args.prediction_filename,
        mistake_filename=args.mistake_filename,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()

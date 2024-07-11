import os
import uuid
from pathlib import Path
import argparse

from lib import check_args, validate_path
from lib import MODEL_FILENAME

current_dir = os.getcwd()
DEFAULT_MODEL_FILE = Path(current_dir + rf"\model\{MODEL_FILENAME}")
DEFAULT_PREDICTED_OUPUT_DIRECTORY = Path(current_dir + r"\predicted")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="RockMinePrediction.py",
        description="The model with ability to predict the Rock or Mine by defined metrics",
    )
    parser.add_argument(
        "-t", "--train", action="store_true", help="switch to train mode"
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="CSV_FILE",
        type=str,
        nargs="?",
        help="predict the classification (Rock or Mine) from dataset",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", metavar="OUTPUT", type=str, nargs="?", help="provide output"
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="PIKLE_FILE",
        type=Path,
        nargs="?",
        help="the model binary file",
        const=DEFAULT_MODEL_FILE,
    )

    return parser


def main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()

    check_args(args, parser)
    if args.train:
        dataset = Path(args.file)
        output = Path(args.output) if args.output else DEFAULT_MODEL_FILE
        if not validate_path(
            dataset,
            "Dataset file does not exists or is not a CSV file",
            file_type=".csv"):
            return

        from core.train import train

        train(dataset, output)
    else:
        input = Path(args.file)
        model = Path(args.model) if args.model else DEFAULT_MODEL_FILE
        output = (
            Path(args.output)
            if args.output
            else DEFAULT_PREDICTED_OUPUT_DIRECTORY.joinpath(str(uuid.uuid4()) + ".csv")
        )
        if not (
            validate_path(
                input,
                "Data to predict does not exists or is not a CSV file",
                file_type=".csv",
            )
            and validate_path(
                model,
                "The model file does not exist or is not a PKL file",
                file_type='.pkl'
            )
        ):
            return

        from core.predict import predict

        predict(model, input, output)
    print("DONE")


if __name__ == "__main__":
    parser = get_parser()
    main(parser)

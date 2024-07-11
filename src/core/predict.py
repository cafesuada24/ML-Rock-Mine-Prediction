from typing import Union, Optional
from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd


def predict(model_path: Path, input: Path, output: Optional[Path] = None) -> pd.Series:
    with open(model_path, 'rb') as f:
        model = load(f)
    data = pd.read_csv(input, header=None)
    pred = pd.Series(model.predict(data), name="Type").map({0: 'M', 1: 'R'})

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(output)
    else:
        print(pred)

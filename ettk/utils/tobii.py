# Built-in Imports
from typing import Dict, Union, Tuple
import pathlib
import gzip
import ast

# Third-party Imports
import pandas as pd
import tqdm

import logging

logger = logging.getLogger(__name__)


def get_absolute_fix(
    gaze_logs: pd.DataFrame, timestamp: Union[int, float], h: int = 1080, w: int = 1920
) -> Tuple[int, int]:

    try:
        raw_fix = (
            gaze_logs[gaze_logs["timestamp"] > timestamp]
            .reset_index()
            .iloc[0]["gaze2d"]
        )
    except IndexError:
        raw_fix = [0, 0]

    if isinstance(raw_fix, str):
        raw_fix = ast.literal_eval(raw_fix)

    fix = (int(raw_fix[0] * w), int(raw_fix[1] * h))

    return fix


def load_g3_file(gz_filepath: pathlib.Path) -> Dict:
    """Load the Tobii g3 file.
    Args:
        gz_filepath (pathlib.Path): The filepath to the gz file.
    Returns:
        Dict: The content of the gz file.
    """
    # Load the data from file
    with open(gz_filepath, mode="rt") as f:
        data = f.read()

    # Convert (false -> False, true -> True, null -> None)
    safe_data = (
        data.replace("false", "False").replace("true", "True").replace("null", "None")
    )

    # Convert the data into dict
    data_dict = ast.literal_eval(safe_data)

    # Return dict
    return data_dict


def load_temporal_gz_file(
    gz_filepath: pathlib.Path, verbose: bool = False
) -> pd.DataFrame:
    """Load the temporal gz file.
    Args:
        gz_filepath (pathlib.Path): Filepath to the gz file.
        verbose (bool): Debugging printout.
    Returns:
        pd.DataFrame: The contents of the gz file.
    """
    # Convert the total_data to a dataFrame
    df = pd.DataFrame()

    # Load the data from file
    with gzip.open(gz_filepath, mode="rt") as f:
        data = f.read()

    data_lines = data.split("\n")

    logger.info(f"Converting {gz_filepath.stem} to .csv for future faster loading.")

    for line in tqdm.tqdm(data_lines, disable=not verbose):

        # Drop empty lines
        if line == "":
            continue

        data_dict = ast.literal_eval(line)
        data = data_dict.pop("data")

        # Skip if the data is missing
        if data == {}:
            continue

        data_dict.update(data)

        insert_df = pd.DataFrame([data_dict])
        df = pd.concat([df, insert_df])

    # Clean the index
    df.reset_index(inplace=True)
    df = df.drop(columns=["index"])

    # Return the data frame
    return df


def load_gaze_data(dir: pathlib.Path, verbose: bool = False) -> pd.DataFrame:
    # Before trying to original data format, check if the faster csv
    # version of the data is available
    gaze_df_path = dir / "gazedata.csv"

    # Loading data, first if csv form, latter with original
    if gaze_df_path.exists():
        gaze_data_df = pd.read_csv(gaze_df_path)
    else:
        gaze_data_df = load_temporal_gz_file(dir / "gazedata.gz", verbose=verbose)
        gaze_data_df.to_csv(gaze_df_path, index=False)

    return gaze_data_df

from pathlib import Path


class Config:
    USE_GPU = True
    DATA_DIR_PATH = Path(__file__).parent.parent.parent.joinpath('data')
    PICKLE_PATH = Path(__file__).parent.joinpath('pickle')

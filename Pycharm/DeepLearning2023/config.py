from pathlib import Path

class Config:
    USE_GPU = False
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_MAX_EPOCH = 100
    DATA_DIR_PATH = Path(__file__).parent.parent.parent.joinpath('data')
    PICKLE_PATH = Path(__file__).parent.joinpath('pickle')

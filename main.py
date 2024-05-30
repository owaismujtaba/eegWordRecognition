import config
import pdb
from pathlib import Path
import numpy as np
from src.utils import laodALlFilesWithPathFromDir
from src.utils import saveFilesAccordingToWord
from src.utils import loadDataWithLables, performPCA
from src.utils import makePCADataset, laodPCADataset
from src.models import fitModels

def main():
    fitModels()


if __name__ == '__main__':

    main()
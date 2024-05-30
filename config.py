import os
from pathlib import Path

curDir = os.getcwd()
dataDir = Path(curDir, 'rawData')
readDir = Path(dataDir, 'Read')
cleanDataDir = Path(curDir, 'cleanDataset')
destinationDatasetDir = Path(curDir, 'Dataset')

print(readDir)
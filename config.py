import os
from pathlib import Path

curDir = os.getcwd()
dataDir = Path(curDir, 'rawData')


readDir = Path(dataDir, 'Read')
cleanDataDir = Path(curDir, 'cleanDataset')
destinationDatasetDir = Path(curDir, 'Dataset')
rawEEGDirectory = Path(dataDir, 'EEG')
imageDataDirectory = Path(curDir, 'ImageData')
COUNT = 0


imageSize  = 256
nClasses = 50
batchSize = 32
epochs=50
import numpy as np
from pathlib import Path
import os
import config
from sklearn.decomposition import PCA
import pdb

def laodALlFilesWithPathFromDir(directory):
    filesWithPaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filesWithPaths.append(os.path.join(root, file))
    return filesWithPaths


def getDestinationPathForFile(filepath):

    filename = filepath.split('\\')[-1]
    word = filepath.split('_')[1]

    folderPath = Path(config.destinationDatasetDir, word)
    os.makedirs(folderPath, exist_ok=True)
    destinationPath = Path(folderPath, filename)

    return destinationPath        

def saveFilesAccordingToWord(filesWithPath):
    for file in filesWithPath:
        fileData = np.load(file)

        destPath = getDestinationPathForFile(file)
        print(f'Saving {destPath}')
        np.save(destPath, fileData)
        

def loadDataWithLables(baseDirectory):
    data = []
    labels = []
    for folder in os.listdir(baseDirectory):
        folder_path = os.path.join(baseDirectory, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(folder_path, file)
                    array = np.load(file_path).flatten()
                    data.append(array)
                    labels.append(folder)

    return np.array(data), np.array(labels)


def performPCA(data, target_variance=0.95):
    pca = PCA()
    pca.fit(data)
    varianceSum = 0.0
    numComponents = 0
    for explainedVariance in pca.explained_variance_ratio_:
        varianceSum += explainedVariance
        numComponents += 1
        if varianceSum >= target_variance:
            break
    
    pca = PCA(n_components=numComponents)
    transformed_data = pca.fit_transform(data)
    return transformed_data



def makePCADataset():
    filePaths = laodALlFilesWithPathFromDir(config.readDir)
    saveFilesAccordingToWord(filePaths)

    data, labels = loadDataWithLables(config.destinationDatasetDir)
    tranformedData = performPCA(data, target_variance=0.95)
    tranformedDataDestinationWithPath = Path(config.cleanDataDir, 'pcaTransformedData.npy')
    labelsDestinationWithPath = Path(config.cleanDataDir, 'labels.npy')
    np.save(tranformedDataDestinationWithPath, tranformedData)
    np.save(labelsDestinationWithPath,  labels)


def laodPCADataset():

    labels = np.load(Path(config.cleanDataDir, 'pcaTransformedData.npy'))
    data = np.load(Path(config.cleanDataDir, 'labels.npy'))
    #pdb.set_trace()
    return labels, data
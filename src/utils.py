import numpy as np
from pathlib import Path
import os
import config
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



def createTrainAndTest(dataGenerator):
    trainData, testData = train_test_split(dataGenerator, test_size=0.2, random_state=42)
    return trainData, testData

def loadImageDatasetFromDirectory(directory, targetSize=(config.imageSize, config.imageSize), batchSize=config.batchSize, classMode='categorical'):
    datagen = ImageDataGenerator(rescale=1./255)  
    return datagen.flow_from_directory(
        directory,
        target_size=targetSize,
        batch_size=batchSize,
        class_mode=classMode
    )

def hasSubDirectories(directory):
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            return True
    return False

def makeDataset(directory):
    for folder in os.listdir(directory):
        folderPath = os.path.join(directory, folder)
        print(folder)

        if not hasSubDirectories(folderPath):
             for file in os.listdir(folderPath):
                filePath = Path(folderPath, file)
                saveFilesAccordingToWord([filePath])
        else:
            print(folderPath)
           
            for subFolder in os.listdir(folderPath):
                print(subFolder)
                subFolderPath = os.path.join(folderPath, subFolder)
                for file in os.listdir(subFolderPath):
                    filePath = Path(subFolderPath, file)
                    saveFilesAccordingToWord([filePath])
              
def getDestinationPathForFile(filepath):

    filename = str(filepath).split('\\')[-1]
    word = str(filepath).split('_')[1]
    folderPath = Path(config.destinationDatasetDir, word)
    os.makedirs(folderPath, exist_ok=True)
    destinationPath = Path(folderPath, filename)
    if os.path.exists(destinationPath):
        word = filename.split('.')
        filename = f'{word[0]}_{config.COUNT}.{word[1]}'
        config.COUNT += 1
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
                    array = np.load(file_path)
                    data.append(array)
                    labels.append(folder)
    
    return np.array(data), np.array(labels)

def plotDataWithLables(baseDirectory):

    for folder in os.listdir(baseDirectory):
        folder_path = os.path.join(baseDirectory, folder)
        print(folder_path)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(folder_path, file)
                    array = np.load(file_path)
                    array = array.reshape(array.shape[1], array.shape[2])
                    plotEEG(array, folder)
                    
    


def plotEEG(data,  folderName):
   
    n_channels, _ = data.shape
    
    folderPath = Path(config.imageDataDirectory, folderName)
    os.makedirs(folderPath, exist_ok=True)
    filename = Path(folderPath, f'{config.COUNT}.png')
    plt.figure(figsize=(10, 100))

    for i in range(n_channels):
        plt.subplot(n_channels, 1, i + 1)
        plt.plot(data[i])
        plt.axis('off')
    config.COUNT += 1  
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    


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

    data = np.load(Path(config.cleanDataDir, 'pcaTransformedData.npy'))
    labels = np.load(Path(config.cleanDataDir, 'labels.npy'))
    return data, labels
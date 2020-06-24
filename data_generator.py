### DATASET GENERATION MODULE: Synthetic Genomic Seq Dataset ##
import pandas as pd
import numpy as np

def createDataset(sampleNumber, featureNumber, balanceRatio, numberOfSpikes, betaglobinIndex):
    # Creates a Pandas Dataframe with shape: (# of samples, # of features) filled with random integers between {0, 1, 2}
    dataset = pd.DataFrame(np.random.randint(0, 3, size=(sampleNumber, featureNumber)))
    # Creates a Pandas Series size of (# of samples) filled with the value 2 (This is supposed to represent HbS)
    betaglobin = [2] * sampleNumber
    # List of "spikes" / enriched alleles
    spikedArray = generateSpikes(sampleNumber, balanceRatio, numberOfSpikes)
    # Assigned label for each sample in a list
    labels = generateLabels(sampleNumber, balanceRatio)
    # "Injects" or inserts the list of spikes to the dataset
    dataset, spikeIndexes = insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex)
    return dataset, labels, spikeIndexes

def generateSpikes(sampleNumber, balanceRatio, numberOfSpikes):
    spikedArray = [] # Initializes empty list (List of spikes)
    variants = [0, 1, 2] # Possible variants represented as counts: ()
    # Probabilities for each of the possible variants to occur based on the corresponding label, in order (0, 1, 2)
    probabilitiesControl = [0.70, 0.15, 0.15]
    probabilitiesDisease = [0.15, 0.15, 0.70]
    # Repeats for the number of spikes desired
    for i in range(0, numberOfSpikes):
        spikedFeature = [] # Empty list (Spiked allele)
        for i in range (0, sampleNumber):
            # If the index is below the cut off sample based on the balance ratio then the sample is in the control cohort, else is in the disease cohort
            if i < sampleNumber * balanceRatio:
        # Inserts the value based on the probabilities above (for Control label)
                spikedFeature.append(np.random.choice(variants, p=probabilitiesControl))
            else:
        # Inserts the value based on the probabilities above (for Disease label)
                spikedFeature.append(np.random.choice(variants, p=probabilitiesDisease))
        # Adds spikes allele to the list
        spikedArray.append(spikedFeature)
    return spikedArray

def generateLabels(sampleNumber, balanceRatio):
    labels = [] # Empty list
    # Iterates through number of samples
    for i in range (0, sampleNumber):
        # If the index is below the cut off sample based on the balance ratio then the sample is in the control cohort, else is in the disease cohort
        if i < sampleNumber * balanceRatio:
            # Control cohort (0)
            labels.append(0)
        else:
            # Disease cohort (1)
            labels.append(1)
    return labels

##### Need to add a way to handle if the spike index has already been used (Possible recursive function? pop from spiked array?)
def insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex):
    spikeIndexes = [] # Empty list that tracks the indexes already used
    dataset[betaglobinIndex] = betaglobin # adds the betaglobin Series to the dataset
    for spike in spikedArray:
        # Selects a random index
        spikeIndex = np.random.randint(0, featureNumber)
        # If the index is not already used by another spike, or the betaglobin index, then proceed
        if spikeIndex != betaglobinIndex and not alreadySpiked(spikeIndex, spikeIndexes):
            # Adds spike to dataset
            dataset[spikeIndex] = spike
            # Adds index to to the tracking array
            spikeIndexes.append(spikeIndex)
    return dataset, spikeIndexes

# Helper function that checks if the spike has already been chosen
def alreadySpiked(index, spikeIndexes):
    for spike in spikeIndexes:
        if spike == index:
            return True

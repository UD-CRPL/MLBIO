### DATASET GENERATION MODULE: Synthetic Genomic Seq Dataset ##
import pandas as pd
import numpy as np

def random_data(size, ratio):
    # Creates random dataset filled with random integers 0, 1 or 2
    dataset = pd.DataFrame(np.random.randint(0, 3, size=size))
    # Creates assigns random labels to each sample on the random datasets
    #labels = np.random.randint(0, 2, size[0])
    labels = generateLabels(size[0], ratio)
    return dataset, labels

# Helper function that checks if the spike has already been chosen
def alreadySpiked(index, spikeIndexes):
    for spike in spikeIndexes:
        if spike == index:
            return True

def generate_spike_indexes(features, number_of_spikes, betaglobin, spike_array):
    index = np.random.randint(0, features)
    if(len(spike_array) >= number_of_spikes):
        return spike_array
    else:
        if(index == betaglobin):
            return generate_spike_indexes(features, number_of_spikes, betaglobin, spike_array)
        elif(alreadySpiked(index, spike_array)):
            return generate_spike_indexes(features, number_of_spikes, betaglobin, spike_array)
        else:
            spike_array.append(index)
            return generate_spike_indexes(features, number_of_spikes, betaglobin, spike_array)

def createDataset(sampleNumber, featureNumber, balanceRatio, numberOfSpikes, betaglobinIndex, spikedArray, spikeIndexes):
    # Creates a Pandas Dataframe with shape: (# of samples, # of features) filled with random integers between {0, 1, 2}
    dataset = pd.DataFrame(np.random.randint(0, 3, size=(sampleNumber, featureNumber)))
    # Creates a Pandas Series size of (# of samples) filled with the value 2 (This is supposed to represent HbS)
    betaglobin = [2] * sampleNumber
    # List of "spikes" / enriched alleles
    if len(spikedArray) == 0:
        spikedArray = generateSpikes(sampleNumber, balanceRatio, numberOfSpikes)
        # Assigned label for each sample in a list
        labels = generateLabels(sampleNumber, balanceRatio)
        # "Injects" or inserts the list of spikes to the dataset
        dataset, spikeIndexes = insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, [])
    else:
        # Assigned label for each sample in a list
        labels = generateLabels(sampleNumber, balanceRatio)
        dataset, spikeIndexes = insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, spikeIndexes)
    return dataset, labels, spikeIndexes, spikedArray

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
    classes = [0, 1]
    probabilities = [1 - balanceRatio, balanceRatio]
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
def insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, spikeIndexes):
    dataset[betaglobinIndex] = betaglobin # adds the betaglobin Series to the dataset
    if(len(spikeIndexes) > 0):
        for i in range(0, len(spikeIndexes)):
            dataset[spikeIndexes[i]] = spikedArray[i]
    else:
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

def create_dataset(samples, features, balance, spikes, betaglobin):
    # Dataframe filled with 0s
    dataset = pd.DataFrame(np.zeros((samples, features)))
    # Fill in each sample
    #dataset = dataset.astype('int32')

    betaglobin_array  = [2.0] * samples
    dataset[betaglobin] = betaglobin_array

    variants = [0, 1, 2] # Possible variants represented as counts: ()
    # Probabilities for each of the possible variants to occur based on the corresponding label, in order (0, 1, 2)
    probabilitiesControl = [0.70, 0.15, 0.15]
    probabilitiesDisease = [0.15, 0.15, 0.70]

    labels = generateLabels(samples, balance)

    spike_indexes = generate_spike_indexes(features, spikes, betaglobin, [])

    for sample_index, sample in dataset.iterrows():
        print(sample_index)
        for column in dataset.columns:
            if(column == betaglobin):
                continue
            elif(alreadySpiked(column, spike_indexes)):
                if(labels[sample_index] == 1):
                    dataset.at[sample_index, column] = np.random.choice(variants, p=probabilitiesDisease)
                elif(labels[sample_index] == 0):
                    dataset.at[sample_index, column] = np.random.choice(variants, p=probabilitiesControl)
            else:
                dataset.at[sample_index, column] = np.random.randint(0, 3)

    return dataset, labels, spike_indexes

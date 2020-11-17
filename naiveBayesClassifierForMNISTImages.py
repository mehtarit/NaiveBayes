import numpy as np
import scipy.io
import math
import geneNewData
from scipy import stats


def imageMeanAndStandardDeviation(sample, meanArray, standardDeviationArray):
    for image in sample:
        avgBrightness = np.mean(image)
        meanArray.append(avgBrightness)
        standardDeviationOfBrightness = np.std(image)
        standardDeviationArray.append(standardDeviationOfBrightness)


def main():
    myID = '6773'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0), len(train1), len(test0), len(test1)])
    print('Your trainset and testset are generated successfully!')

    Feature1TrainingList0 = []
    Feature1TrainingList1 = []
    Feature2TrainingList0 = []
    Feature2TrainingList1 = []

    Feature1TestingList0 = []
    Feature1TestingList1 = []
    Feature2TestingList0 = []
    Feature2TestingList1 = []

    # Task1
    imageMeanAndStandardDeviation(
        train0, Feature1TrainingList0, Feature2TrainingList0)
    imageMeanAndStandardDeviation(
        train1, Feature1TrainingList1, Feature2TrainingList1)

    # Doing this for testing set to use in Task3
    imageMeanAndStandardDeviation(
        test0, Feature1TestingList0, Feature2TestingList0)
    imageMeanAndStandardDeviation(
        test1, Feature1TestingList1, Feature2TestingList1)

    # Task2
    meanFeature1Train0 = np.mean(Feature1TrainingList0)
    varianceFeature1Train0 = np.var(Feature1TrainingList0)
    meanFeature2Train0 = np.mean(Feature2TrainingList0)
    varianceFeature2Train0 = np.var(Feature2TrainingList0)

    print('Mean of feature1 for digit0', meanFeature1Train0)
    print('Variance of feature1 for digit0', varianceFeature1Train0)
    print('Mean of feature2 for digit0', meanFeature2Train0)
    print('Variance of feature2 for digit0', varianceFeature2Train0)

    meanFeature1Train1 = np.mean(Feature1TrainingList1)
    varianceFeature1Train1 = np.var(Feature1TrainingList1)
    meanFeature2Train1 = np.mean(Feature2TrainingList1)
    varianceFeature2Train1 = np.var(Feature2TrainingList1)

    print('Mean of feature1 for digit1', meanFeature1Train1)
    print('Variance of feature1 for digit1', varianceFeature1Train1)
    print('Mean of feature2 for digit1', meanFeature2Train1)
    print('Variance of feature2 for digit1', varianceFeature2Train1)

    #Task3 : Predict

    # since 0 and 1 are the only possible prediction outcomes there are 2 classes
    possibleClassesCount = 2
    sampleCountTest0 = len(test0)
    sampleCountTest1 = len(test1)
    predictionMatrixTest0 = np.zeros((sampleCountTest0, possibleClassesCount))
    predictionMatrixTest1 = np.zeros((sampleCountTest1, possibleClassesCount))

    priorProbabilityOfDigit0 = 0.5
    priorProbabilityOfDigit1 = 0.5

    feature1Train0NormalDistribution = stats.norm(
        meanFeature1Train0, math.sqrt(varianceFeature1Train0))
    feature2Train0NormalDistribution = stats.norm(
        meanFeature2Train0, math.sqrt(varianceFeature2Train0))

    feature1Train1NormalDistribution = stats.norm(
        meanFeature1Train1, math.sqrt(varianceFeature1Train1))
    feature2Train1NormalDistribution = stats.norm(
        meanFeature2Train1, math.sqrt(varianceFeature2Train1))

    for i in range(sampleCountTest0):

        probabilityOfImageFeature1GivenDigit0 = feature1Train0NormalDistribution.pdf(
            Feature1TestingList0[i])
        probabilityOfImageFeature2GivenDigit0 = feature2Train0NormalDistribution.pdf(
            Feature2TestingList0[i])
        probabilityOfImageFeature1GivenDigit1 = feature1Train1NormalDistribution.pdf(
            Feature1TestingList0[i])
        probabilityOfImageFeature2GivenDigit1 = feature2Train1NormalDistribution.pdf(
            Feature2TestingList0[i])

        likelyhoodOfDigit0 = probabilityOfImageFeature1GivenDigit0 * \
            probabilityOfImageFeature2GivenDigit0*priorProbabilityOfDigit0
        likelyhoodOfDigit1 = probabilityOfImageFeature1GivenDigit1 * \
            probabilityOfImageFeature2GivenDigit1*priorProbabilityOfDigit1

        # we dont care about the denominators
        predictionMatrixTest0[i, 0] = likelyhoodOfDigit0
        # we dont care about the denominators since those are the same
        predictionMatrixTest0[i, 1] = likelyhoodOfDigit1

    for i in range(sampleCountTest1):

        probabilityOfImageFeature1GivenDigit0 = feature1Train0NormalDistribution.pdf(
            Feature1TestingList1[i])
        probabilityOfImageFeature2GivenDigit0 = feature2Train0NormalDistribution.pdf(
            Feature2TestingList1[i])
        probabilityOfImageFeature1GivenDigit1 = feature1Train1NormalDistribution.pdf(
            Feature1TestingList1[i])
        probabilityOfImageFeature2GivenDigit1 = feature2Train1NormalDistribution.pdf(
            Feature2TestingList1[i])

        likelyhoodOfDigit0 = probabilityOfImageFeature1GivenDigit0 * \
            probabilityOfImageFeature2GivenDigit0*priorProbabilityOfDigit0
        likelyhoodOfDigit1 = probabilityOfImageFeature1GivenDigit1 * \
            probabilityOfImageFeature2GivenDigit1*priorProbabilityOfDigit1

        # we dont care about the denominators
        predictionMatrixTest1[i, 0] = likelyhoodOfDigit0
        # we dont care about the denominators since those are the same
        predictionMatrixTest1[i, 1] = likelyhoodOfDigit1

    # figure out the classification from prediction matrix

    actualClassificationMatrixTest0 = np.argmax(predictionMatrixTest0, axis=1)
    actualClassificationMatrixTest1 = np.argmax(predictionMatrixTest1, axis=1)

    # Task4 : Calculate the accuracy
    expectedClassificationMatrixTest0 = np.zeros(len(test0))
    expectedClassificationMatrixTest1 = np.ones(len(test1))

    accuracyTest0 = np.mean(actualClassificationMatrixTest0 ==
                            expectedClassificationMatrixTest0)
    accuracyTest1 = np.mean(actualClassificationMatrixTest1 ==
                            expectedClassificationMatrixTest1)

    print('Accuracy for test0: ', accuracyTest0)
    print('Accuracy for test1: ', accuracyTest1)


if __name__ == '__main__':
    main()

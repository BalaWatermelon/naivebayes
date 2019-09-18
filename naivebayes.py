# TODO: Create unit test
# Reduce magic numbers
MEAN = 0
STD = 1
CLASS = 0


def readData(filename):
    dataset = []
    with open(filename) as file:
        for line in file.readlines():
            dataset.append([int(x) for x in line.strip().split(' ')])
    return dataset


def buildTrainData(dataset, label):
    # dataset = [['1','2'],['1','2']]
    # label = [['1','0'],['CLASS','ROWID']]
    """
    >>> buildTrainData([[1,2],[3,4]],[[1,0],[1,0]]) == {1:[[1,2],[3,4]]}
    True

    >>> buildTrainData([[1,2],[3,4],[1,2]],[[1,0],[0,1],[3,2]]) == {1:[[1,2]],0:[[3,4]],3:[[1,2]]}
    True

    """
    trainData = {}
    for data, label in zip(dataset, label):
        if label[CLASS] in trainData:
            trainData[label[CLASS]].append(data)
        else:
            trainData[label[CLASS]] = [data]
    return trainData


def mean(array):
    """
    >>> mean([1,2,3])
    2.0
    >>> mean([3,3,4,4,5,6])
    4.1666667
    """
    mean = 0
    for num in array:
        mean += num
    return round(mean/len(array), 7)


def standardD(array):
    m = mean(array)
    distances = [(n-m)**2 for n in array]
    # TODO: Sample or normal variance?
    variance = sum(distances)/(len(distances)-1)
    return variance**0.5


def classWeights(trainData):
    # trainData = {'0':[[0,2],[0,1],[1,2]],'1':[[44,50],[45,50],[44,51]]}
    weightsData = {}
    for classLabel in trainData:
        weights = [(mean(classColumn), standardD(classColumn))
                   for classColumn in zip(*trainData[classLabel])]
        weightsData[classLabel] = weights
    return weightsData


def predict(row, weightData):
    # row = ['2','3']
    # weightData = {'0':[(mean, std),(mean, std)],
    #               '1':[(mean, std),(mean, std)}
    minDistance = None
    resultClass = None
    for classLabel in weightData:
        classDistance = sum([((feature - weight[MEAN]) / weight[STD])
                             ** 2 for feature, weight in zip(row, weightData[classLabel])])
        if minDistance == None or classDistance < minDistance:
            minDistance = classDistance
            resultClass = classLabel
    return resultClass


def main():
    dataSet = readData('testBayes.data')
    label = readData('testBayes.trainlabels.0')
    trainData = buildTrainData(dataSet, label)
    trainData = {'0': [[0, 2], [0, 1], [1, 2]],
                 '1': [[44, 50], [45, 50], [44, 51]]}
    weightData = classWeights(trainData)
    testSet = readData('test.data')
    rowId = 0
    for row in testSet:
        result = predict(row, weightData)
        print(row, result, rowId)
        rowId += 1


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

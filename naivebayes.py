# Small mean margin
MARGIN = 0.01
# Reduce magic numbers
MEAN = 0
STD = 1
CLASS = 0


def readData(filename):
    dataset = []
    with open(filename) as file:
        for line in file.readlines():
            dataset.append([float(x) for x in line.strip().split(' ')])
    return dataset


def readLabel(filename):
    labelSet = {}
    with open(filename) as file:
        for line in file.readlines():
            klass, rowId = line.strip().split()
            klass = int(klass)
            rowId = int(rowId)
            labelSet[rowId] = klass
    return labelSet


def buildTrainData(dataset, labelSet):
    trainData = {}
    predictData = {}
    for rowId in range(len(dataset)):
        if rowId in labelSet:
            # This is a row with label
            if labelSet[rowId] not in trainData:
                trainData[labelSet[rowId]] = [dataset[rowId]]
            else:
                trainData[labelSet[rowId]].append(dataset[rowId])
        else:
            # This is a predict row, record rowId and value
            predictData[rowId] = dataset[rowId]
    return trainData, predictData


def mean(array):
    mean = MARGIN
    for num in array:
        mean += num
    return round(mean/float(len(array)), 7)


def standardD(array):
    m = mean(array)
    distances = [(n-m)**2 for n in array]
    variance = sum(distances)/float(len(distances))
    if variance == 0:
        array = list(array)
        array.append(m-0.1)
        return standardD(array)
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
    import sys
    if len(sys.argv) != 3:
        print('Wrong number of arguments, exiting...')
        sys.exit(1)
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    dataSet = readData(dataFile)
    labelSet = readLabel(labelFile)
    trainData, predictData = buildTrainData(dataSet, labelSet)
    weightData = classWeights(trainData)
    for key in predictData:
        result = predict(predictData[key], weightData)
        print(f'{result} {key}')


if __name__ == "__main__":
    main()

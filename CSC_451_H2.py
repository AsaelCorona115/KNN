import numpy as np
from sklearn import neighbors


# Reads the data from the files and outputs them into arrays
def file_to_array(path):
    # Reading a file and converting it into an usable array using readline()
    file1 = open(path, 'r')
    count = 0
    resulting_array = []
    while True:
        count += 1
        single_point = []
        # Get next line from file
        line = file1.readline().strip()

        # if line is empty
        # end of file is reached
        if not line:
            break
        getting_points = line.split(',')
        single_point.append(float(getting_points[0]))
        single_point.append(float(getting_points[1]))
        resulting_array.append(single_point)

    file1.close()
    return resulting_array

# Takes the arrays from the function above and calculates the features


def data_extraction(unprocessed_data):
    # Defining an empty array where to deposit all the extracted data
    extracted_data = []

    # All required points to calculate the features
    point1 = np.array(unprocessed_data[0])
    point2 = np.array(unprocessed_data[1])
    point3 = np.array(unprocessed_data[2])
    point4 = np.array(unprocessed_data[3])
    point5 = np.array(unprocessed_data[4])
    point6 = np.array(unprocessed_data[5])
    point7 = np.array(unprocessed_data[6])
    point8 = np.array(unprocessed_data[7])
    point9 = np.array(unprocessed_data[8])
    point10 = np.array(unprocessed_data[9])
    point11 = np.array(unprocessed_data[10])
    point12 = np.array(unprocessed_data[11])
    point13 = np.array(unprocessed_data[12])
    point15 = np.array(unprocessed_data[14])
    point16 = np.array(unprocessed_data[15])
    point17 = np.array(unprocessed_data[16])
    point18 = np.array(unprocessed_data[17])
    point19 = np.array(unprocessed_data[18])
    point20 = np.array(unprocessed_data[19])
    point21 = np.array(unprocessed_data[20])
    # Necessary calculations to calculate some features

    # Calculations necessary for eye_length_Ratio
    left_eye_length = np.linalg.norm(point9-point10)
    right_eye_length = np.linalg.norm(point11-point12)

    if left_eye_length > right_eye_length:
        max_eye_length = left_eye_length
    else:
        max_eye_length = right_eye_length

    # Calculations necessary for eye_brow_length_Ratio
    between_4_and_5 = np.linalg.norm(point4 - point5)
    between_8_and_13 = np.linalg.norm(point8 - point13)

    if between_8_and_13 > between_4_and_5:
        max_of_two = between_8_and_13
    else:
        max_of_two = between_4_and_5

    # Feature calculation

    eye_length_ratio = max_eye_length / np.linalg.norm(point8 - point13)
    eye_distance_ratio = np.linalg.norm(
        point8 - point1) / np.linalg.norm(point8 - point13)
    nose_ratio = np.linalg.norm(point15 - point16) / \
        np.linalg.norm(point20 - point21)
    lip_size_ratio = np.linalg.norm(
        point2 - point3) / np.linalg.norm(point17 - point18)
    lip_length_ratio = np.linalg.norm(
        point2 - point3) / np.linalg.norm(point20 - point21)
    eye_brow_length_Ratio = max_of_two / np.linalg.norm(point8 - point13)
    aggresive_ratio = np.linalg.norm(
        point10 - point19) / np.linalg.norm(point20 - point21)
    # Pushing to the array of the substracted data
    extracted_data.append(eye_length_ratio)  # Extracted data is in position 0
    extracted_data.append(eye_distance_ratio)
    extracted_data.append(nose_ratio)
    extracted_data.append(lip_size_ratio)
    extracted_data.append(lip_length_ratio)
    extracted_data.append(eye_brow_length_Ratio)
    extracted_data.append(aggresive_ratio)

    # Returning the resulting array
    return extracted_data

# Calculates Confusion Matrix, precision, recall rate and accuracy


def binaryClassifier_evaluator(predicted_Classes, trueClasses):
    # array containing the accuracy, precision and the recall rate
    evaluation = []
    # Counters
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    # Calculating the data above
    for i in range(len(trueClasses)):
        correctClass = trueClasses[i]
        if((correctClass == 0) and (predicted_Classes[i] == 0)):
            truePositive = truePositive + 1
        elif((correctClass == 0) and (predicted_Classes[i] == 1)):
            falseNegative = falseNegative + 1
        elif((correctClass == 1) and (predicted_Classes[i] == 0)):
            falsePositive = falsePositive + 1
        else:
            trueNegative = trueNegative + 1

    # Calculating Accuracy, Precision, Recall Rate
    accuracy = (truePositive + trueNegative) / len(trueClasses)
    precision = (truePositive) / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)

    # appending the results
    evaluation.append(accuracy)
    evaluation.append(precision)
    evaluation.append(recall)

    return evaluation


# Training Range
TrainingSubfolderIndex = range(1, 4)

# Testing Range
TestingSubfolderIndex = range(4, 6)

# the file number for testing is the same for training and testing
FileNumberIndex = range(1, 5)

# Creating Empty Arrays for the extracted data
trainingData = []  # This array will include feature data, no classes
trainingClasses = []  # This array will include the classes of the training data
testingData = []  # This array will include the data of the features
testingClasses = []  # This array will include the classes of the testing data

# Reading the male files and appending to the training data
for i in TrainingSubfolderIndex:
    for j in FileNumberIndex:
        data = file_to_array('Face_Database_Arrays\m-00' +
                             str(i) + '\m-00' + str(i) + '-0' + str(j) + '.pts')
        result = data_extraction(data)
        trainingData.append(result)
        trainingClasses.append(0)


# Reading the female files and appending to the training data
for i in TrainingSubfolderIndex:
    for j in FileNumberIndex:
        data = file_to_array('Face_Database_Arrays\w-00' +
                             str(i) + '\w-00' + str(i) + '-0' + str(j) + '.pts')
        result = data_extraction(data)
        trainingData.append(result)
        trainingClasses.append(1)

# Reading the male files and appending to the testing data
for i in TestingSubfolderIndex:
    for j in FileNumberIndex:
        data = file_to_array('Face_Database_Arrays\m-00' +
                             str(i) + '\m-00' + str(i) + '-0' + str(j) + '.pts')
        result = data_extraction(data)
        testingData.append(result)
        testingClasses.append(0)

# Reading the female files and appending to the testing data
for i in TestingSubfolderIndex:
    for j in FileNumberIndex:
        data = file_to_array('Face_Database_Arrays\w-00' +
                             str(i) + '\w-00' + str(i) + '-0' + str(j) + '.pts')
        result = data_extraction(data)
        testingData.append(result)
        testingClasses.append(1)

# Training Data output
# print(trainingData)
# print(len(trainingData))
# print(trainingClasses)
# print(len(trainingClasses))

# Testing Data output
# print(testingData)
# print(len(testingData))
# print(testingClasses)
# print(len(testingClasses))


# The following loop runs the algorithm and calculates the evaluation values for k from 1-15
for i in range(1, 11):
    nn = neighbors.KNeighborsClassifier(i)
    nn.fit(trainingData, trainingClasses)  # Training the algorithm
    predictions = nn.predict(testingData)  # Calculating predictions

    # The returned array returns: accuracy, precision, recall rate in order in the array
    evaluation = binaryClassifier_evaluator(predictions, testingClasses)
    print('k is ' + str(i) + ', Accuracy: ' + str(evaluation[0]) + ', Precision: ' + str(
        evaluation[1]) + ', Recall Rate:  ' + str(evaluation[2]))

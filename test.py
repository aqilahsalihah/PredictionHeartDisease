import joblib
import numpy as np

# Load the trained model
modelNN = joblib.load('predictorNN.joblib')
# modelLR = joblib.load('predictorLR.joblib')
# modelSVM = joblib.load('predictorSVM.joblib')
# modelGSNB = joblib.load('predictorGSNB.joblib')

# 58	M	ATA	136	164	0	ST	99	Y	2	Flat	1
# 53	M	ASY	124	260	0	ST	112	Y	3	Flat	0


# Sample input data
input_data = np.array([58, 1, 1, 136, 164, 0, 1, 99, 1, 2, 1]).reshape(1, -1)

input_data2 = [53, 'Male', 'Asymptomatic', 124, 260, 'No', 'Normal', 112, 'Yes', 3.0, 'Flat']

# Create a dictionary to map the input values to the corresponding label encodings
value_mappings = {
    'Sex': {'Female': 0, 'Male': 1},
    'ChestPainType': {'Asymptomatic': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Typical Agina': 3},
    'FastingBS': {'No': 0, 'Yes': 1},
    'RestingECG': {'LVH': 0, 'Normal': 1, 'ST': 2},
    'ExerciseAngina': {'No': 0, 'Yes': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}

# Reformat the input data
reformatted_data = []
for feature, value in zip(['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'], input_data2):
    if feature in value_mappings:
        encoded_value = value_mappings[feature].get(value)
        reformatted_data.append(encoded_value)
    else:
        reformatted_data.append(value)

# Convert the reformatted data to the desired format
input_data_array = np.array(reformatted_data).reshape(1, -1)

# Display the reformatted input data
print(input_data2)
print(input_data_array)



# Make predictions
predictionsNN1 = modelNN.predict(input_data)
# predictionsLR1 = modelLR.predict(input_data)
# predictionsSVM1 = modelSVM.predict(input_data)
# predictionsGSNB1 = modelGSNB.predict(input_data)

predictionsNN2 = modelNN.predict(input_data_array)
# predictionsLR2 = modelLR.predict(input_data_array)
# predictionsSVM2 = modelSVM.predict(input_data_array)
# predictionsGSNB2 = modelGSNB.predict(input_data_array)

# Display the predictions
# print('data 1: ')
# print('NN', predictionsNN1)
# print('LR', predictionsLR1)
# print('SVM', predictionsSVM1)
# print('GSNM', predictionsGSNB1)

# print()
# print('data 2: ')
# print('NN', np.round(predictionsNN2))
# print('LR', predictionsLR2)
# print('SVM', predictionsSVM2)
# print('GSNM', predictionsGSNB2)

rounded_predictionsNN2 = np.round(predictionsNN2)

if rounded_predictionsNN2 == 0:
    prediction_label = 'No heart disease'
else:
    prediction_label = 'Heart disease'

print('NN2 (expected 0)', prediction_label)


rounded_predictionsNN2 = np.round(predictionsNN1)

if rounded_predictionsNN2 == 0:
    prediction_label = 'No heart disease'
else:
    prediction_label = 'Heart disease'

print('NN1 (expected 1)', prediction_label)

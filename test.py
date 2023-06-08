# Testing Model


import numpy as np
from tensorflow import keras

# Load the trained model
modelNN = keras.models.load_model('predictoNN.h5')

# 58	M	ATA	136	164	0	ST	99	Y	2	Flat	1
# 53	M	ASY	124	260	0	ST	112	Y	3	Flat	0


# Sample input data
input_data = np.array([58, 1, 1, 136, 164, 0, 1, 99, 1, 2, 1]).reshape(1, -1)
input_data1 = [58, 'Male', 'Atypical Angina', 136, 164, 'No', 'Normal', 99, 'Yes', 2.0, 'Flat'] #HD
input_data2 = [30, 'Male', 'Typical Agina', 0, 200, 'Yes', 'Normal', 60, 'No', 1.0, 'Up'] #NHD


# Create a dictionary to map the input values to the corresponding label encodings
value_mappings = {
    'Sex': {'Female': 0, 'Male': 1},
    'ChestPainType': {'Asymptomatic': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Typical Agina': 3},
    'FastingBS': {'No': 0, 'Yes': 1},
    'RestingECG': {'LVH': 0, 'Normal': 1, 'ST': 2},
    'ExerciseAngina': {'No': 0, 'Yes': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}


# Reformat the input data1
reformatted_data1 = []
for feature, value in zip(['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'], input_data1):
    if feature in value_mappings:
        encoded_value = value_mappings[feature].get(value)
        reformatted_data1.append(encoded_value)
    else:
        reformatted_data1.append(value)

# Convert the reformatted data to the desired format
input_data_array1 = np.array(reformatted_data1).reshape(1, -1)




# Reformat the input data2
reformatted_data2 = []
for feature, value in zip(['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'], input_data2):
    if feature in value_mappings:
        encoded_value = value_mappings[feature].get(value)
        reformatted_data2.append(encoded_value)
    else:
        reformatted_data2.append(value)

# Convert the reformatted data to the desired format
input_data_array2 = np.array(reformatted_data2).reshape(1, -1)





# DATA1
print('DATA 1 expected HD')
print(input_data1)
print(input_data_array2)
predictionsNN1 = modelNN.predict(input_data_array1)
print(predictionsNN1)

rounded_predictionsNN = np.round(predictionsNN1)

if rounded_predictionsNN == 0:
    prediction_label = 'No heart disease'
else:
    prediction_label = 'Heart disease'

print('NN1 (expected 1)', prediction_label)



# DATA2
print()
print('DATA 2 expected NHD')
print(input_data2)
print(input_data_array2)
predictionsNN2 = modelNN.predict(input_data_array2)
print(predictionsNN2)

rounded_predictionsNN = np.round(predictionsNN2)

if rounded_predictionsNN == 0:
    prediction_label = 'No heart disease'
else:
    prediction_label = 'Heart disease'

print('NN2 (expected 0)', prediction_label)

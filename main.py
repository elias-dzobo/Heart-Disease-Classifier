import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC

data = pd.read_csv('framingham.csv')
print(data.head())
print(data.columns)

col_missing = [col for col in data.columns if data[col].isnull().any()]
#print(col_missing)

#finding relationship between features
glucose = data['glucose']
BMI = data['BMI']

#m, b = np.polyfit(BMI, glucose, 1)
#print('M:', m, '-----B:', b)

#plt.scatter(glucose, BMI, alpha=0.5, c=BMI, cmap='Spectral')
#plt.plot(cigPerDay, m*cigPerDay + BMI)
sns.lmplot(x='cigsPerDay', y='sysBP', data=data)
#plt.show()

data.dropna(axis=0, inplace=True)

#feature Extraction
x = data[['age', 'currentSmoker', 'cigsPerDay', 'prevalentStroke', 'diabetes', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose']]
y = data['TenYearCHD']

columns = x.columns
#cleaning data
#print(data.isnull().values.any())

imputer = SimpleImputer()

#splitting data and building model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

new_train  = pd.DataFrame(imputer.fit_transform(x_train))
new_test = pd.DataFrame(imputer.transform(x_test))

new_train.columns = x_train.columns
new_test.columns = x_test.columns

model = LogisticRegression()
model.fit(new_train, y_train)

score = model.score(new_test, y_test)
print(score)

predictions = model.predict(new_test)
#print(predictions)

new_data = [50, 1, 45, 0, 0, 300.0, 150.0, 75, 80, 90]

df = pd.DataFrame({'age':[50], 'currentSmoker':[1], 'cigsPerDay':[45], 'prevalentStroke':[0], 'diabetes': [0], 'totChol':[300.0], 'sysBP':[150.0], 'BMI':[75], 'heartRate': [80], 'glucose': [90]})
print(df)


coef = model.coef_
coef_list = np.ndarray.tolist(coef)
print(coef_list)

list_coef = [j for sub in coef_list for j in sub]

zipped_list = list(zip(columns, list_coef))
print(zipped_list)
# feature_0_score = max(list_coef)
# idx = list_coef.index(feature_0_score)

# feature_0 = columns[idx]
# print('first essential feature: ', feature_0)

# list_coef.remove(feature_0_score)
# feature_1_score = max(list_coef)
# idx_1 = list_coef.index(feature_1_score)

# feature_1 = columns[idx_1]
# print('Second essential feature: ', feature_1)

new_x = data[['diabetes', 'prevalentStroke']]
new_y = data['TenYearCHD']

new_train_x, new_test_x, new_train_y, new_test_y = train_test_split(new_x, new_y, test_size=0.2, random_state=1)

final_test_x = pd.DataFrame(new_test_x)
final_test_y = pd.DataFrame(new_test_y)

final_train_x = pd.DataFrame(new_train_x)
final_train_y = pd.DataFrame(new_train_y)

new_model = LogisticRegression()

new_model.fit(final_train_x, final_train_y.values.ravel())
new_score = new_model.score(final_test_x, final_test_y.values.ravel())
print('New score is: ', new_score*100, '%')
print(new_score > score)

def predict(df):
    prediction = new_model.predict(df)
    if prediction[0] == 0:
        return 'No risk Of heart disease'
    else:
        return 'Risk of heart disease'

new_df = pd.DataFrame({'age':[65], 'prevalentStroke':[1]})
print(predict(new_df))
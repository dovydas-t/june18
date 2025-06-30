import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, root_mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("june18/personality_dataset.csv")

print(df.head())
print(df.shape)

# # Count NaN values in each column
# nan_counts = df.isna().sum()
# print(nan_counts)

# sns.heatmap(df.isna(), cbar=False)
# plt.show()

df_cleaned = df.dropna()

print(df_cleaned.shape)


target = df_cleaned["Personality"]
data = df_cleaned.drop(columns="Personality")

stage_fear = data['Stage_fear']
das = data['Drained_after_socializing']



encoder_target = LabelEncoder()
encoder_stage_fear = LabelEncoder()
encoder_das = LabelEncoder()

encoder_target.fit(target)
targets_encoded = encoder_target.transform(target)

encoder_stage_fear.fit(stage_fear)
stage_fear_encoded = encoder_stage_fear.transform(stage_fear)

encoder_das.fit(das)
das_encoded = encoder_das.transform(das)


# # print(targets_encoded)
# # print(encoder_label.inverse_transform(targets_encoded))
# # print(stage_fear_encoded)
# # print(encoder_label.inverse_transform(stage_fear_encoded))
# # print(das_encoded)
# # print(encoder_label.inverse_transform(das_encoded))


data['Stage_fear'] = stage_fear_encoded
data['Drained_after_socializing'] = das_encoded



# # print(data.head())



# Rename to avoid overwriting
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(data, target, test_size=0.35, random_state=42)


# # print(x_train)
# # print(y_train)

# # print(x_test)
# # print(y_test)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_clf)
x_test_scaled = scaler.transform(x_test_clf)



model_clasification_k = KNeighborsClassifier(n_neighbors=7)
model_clasification_k.fit(x_train_scaled, y_train_clf)

y_pred_clf = model_clasification_k.predict(x_test_scaled)

report = classification_report(y_test_clf, y_pred_clf)
print(report)








x_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=30, random_state=42)

linear_model = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

linear_score = cross_val_score(linear_model, x_reg, y_reg, cv=kfold, scoring="r2")

print(linear_score)
print(np.average(linear_score))



x_train, x_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=0.3, random_state=42)
model = LinearRegression()


model.fit(x_train,y_train)

Y_pred = model.predict(x_test)

rmse = root_mean_squared_error(y_test, Y_pred)
r2 = r2_score(y_test, Y_pred)

print("RMSE: ", rmse) # 0 perfect more means more error
print("r2: ", r2) # 0-1 - 1 means perfrect





plt.scatter(y_test, Y_pred)
plt.xlabel("Tiesa")
plt.ylabel("Spėjimas")
plt.show()



# Now compute the confusion matrix correctly
cm = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder_target.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Load of libraries

# %%
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# %% [markdown]
# # 📜 About this dataset
# ## Feature description
# + **Age**: age of the patient [years]
# + **Sex**: sex of the patient [M: Male, F: Female]
# + **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# + **RestingBP**: resting blood pressure [mm Hg]
# + **Cholesterol**: serum cholesterol [mm/dl]
# + **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# + **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# + **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
# + **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]
# + **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
# + **ST_Slope**: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# + **HeartDisease**: output class [1: heart disease, 0: Normal]

# %%
df = pd.read_csv('D:\Education\Project\Project\Heart_Disease_Prediction_Using_Machine_Learning\heart.csv')
df.head()

# %%
df.shape

# %%
df.drop_duplicates()

# %%
df.info()

# %%
continuos_f = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_f = ["ChestPainType", "RestingECG", "ST_Slope"]
binaries_f = ["Sex", "FastingBS", "ExerciseAngina"]

# %%
df.isna().all()

# %%
df[continuos_f].describe()

# %% [markdown]
# ****
# 
# # 📊 EDA and data wrangling 

# %%
plt.style.use("seaborn")
plt.subplots_adjust(hspace=0.2)
color = 'winter'

fig, axs = plt.subplots(6, 2, figsize=(15,28))
i=1
for feature in df.columns:
    if feature not in ["HeartDisease"] and i < 14:
        plt.subplot(6,2,i)
        sns.histplot(data=df, x=feature, kde=True, palette=color, hue='HeartDisease')
        i+=1

# %% [markdown]
#  Based on this visualization, we will an analysis:
#  + **Age**: We see how the risk of suffering cardiovascular diseases (CVDs) increases with age.
#  + **Sex**: Apparently the men have a major risk than women of suffering CVDs.
#  + **ChestPainType**: The majority cases of CVDs present absence of chest pain or the usual anginal equivalents. Followed by the cases that present chest pain non-anginal.
#  + **RestingBP**: According to the [heart.org's article](https://www.heart.org/en/health-topics/high-blood-pressure/health-threats-from-high-blood-pressure/how-high-blood-pressure-can-lead-to-heart-failure), the high blood pressure can drive to suffering CVDs. We see in this histogram a slight increase in the CVDs cases when increase the resting blood pressure. <span style="color:red; font-weight: bold;">Let's observe that exists zero values of blood pressure, this it is incorrect we must treat them</span>.
#  + **Cholesterol**: It is well known that a high level of cholesterol can lead to stroke, heart attack, or even heart failure. However, according to the [Heatline's article](https://www.healthline.com/health/serum-cholesterol) not all serum cholesterol is bad, exist three cholesterol types that compose a serum cholesterol, HDL or 'good' cholesterol, LDL or 'bad' cholesterol and triglycerides(TG), where high HDL levels are better than high LDL and TG levels. The level of serum cholesterol (SC) can be calculated like the sum between levels of HDL, LDL and TG, SC[mm/dl] = HDL[mm/dl] + LDL[mm/dl] + TG[mm/dl], note that we don't know the concentrations of HDL, LDL and TG in serum cholesterol of our dataset, but Heatline get us a reference level to determinate when a pacient have risk of suffering CDVs. A serum cholesterol major than 200 mm/dl is considered a risk factor for health, let'us in our histogram that majority cases of CVDs have major levels that 200 mm/dl. In other hand, <span style="color:red; font-weight: bold;">let's observe that exists zero values and value too high (atypical cases) of serum cholesterol, this it is incorrect we must treat them</span>.
#  + **FastingBS**: According to [CDC's article](https://www.cdc.gov/diabetes/library/features/diabetes-and-heart.html#:~:text=Over%20time%2C%20high%20blood%20sugar,and%20can%20damage%20artery%20walls.), high blood sugar can damage blood vessels and the nerves that control your heart. For this reason is that patients with diabetes have most risk of suffering CVDs. Our dataset reflect this condition, where the majority of patients with high blood sugar have CVDs.
#  + **RestingECG**: The majority cases of patients with CVDs present normal resting electrocardiograms, but we observe that in the cases that present ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) the amount of patients with CVDs are two times more than patientes that not suffering CVDs, something similar occurs with cases that present probable or definite left ventricular hypertrophy but to a lesser degree.
#  + **MaxHR**: we don't know the conditions in which they were measured this values, so we can't talk about if this values can be to correct. The truth is that patients with CVDs present low values of maximum heart rate, while that patients with normal conditions present values more high.
#  + **ExerciseAngina**: Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood. But angina is not a disease. It's a symptom of an underlying heart problem, and for this reason that the majority of patients with CVDs present exercise-induced angina.
#  + **Oldpeak**: Also called ST segment depression is a factor that relationed with several CVD, but in some cases this depression can be normal. We can observe that patients with CVDs present high values of depression, though also some patients with CVDs present zero values of depression.
#  + **ST_slope**: According to [Ecgwaves's article](https://ecgwaves.com/topic/ecg-st-segment-depression-ischemia-infarction-differential-diagnoses/), upsloping ST segment can be generally considered like normal, but so much flat or horizontal ST segment and dowsloping ST segment can be considered like anomalies, and they are present in the majority cases of patients with CVDs as reflected by our histogram.
#  

# %% [markdown]
# ### Outlier Detection
# 
# We have previously seen that features like cholesterol and resting blood pressure present atypical cases, this cases not represent the condition of general population and this data type we don't useful to train our model of predictions, let's see how to treat them:

# %%
def detect_outliers(label=None):
    Q1 = df[label].quantile(0.25)
    Q3 = df[label].quantile(0.75)
    IQR = Q3 - Q1
    interval = ((df[label] > Q1 - 1.5*IQR) & (df[label] < Q3 + 1.5*IQR))
    return df[interval], df[~interval]

def assign_mean(df_out, not_df_out, label=None):
    df.loc[df_out[df_out["HeartDisease"] == 0].index, label] = not_df_out[not_df_out["HeartDisease"] == 0][label].mean()
    df.loc[df_out[df_out["HeartDisease"] == 1].index, label] = not_df_out[not_df_out["HeartDisease"] == 1][label].mean()
    return

def delete_outliers(df_out):
    return df.drop(df_out.index)

# %% [markdown]
# #### Cholesterol

# %%
plt.figure(figsize=(20,5))
sns.boxplot(data=df, x="Cholesterol")

# %%
not_df_out_ch, df_out_ch = detect_outliers('Cholesterol')
print(f'Outliers in cholesterol represent the {round((df_out_ch.shape[0]*100)/df.shape[0], 2)}% of our dataset')
df_out_ch

# %% [markdown]
# These outliers represent the 19.93% of our dataset. There are registers with zero values these are human errors, the best solution consist is to deleted them while the rest of the register will be assigned the mean cholesterol of the data set.

# %%
df = delete_outliers(df_out_ch[df_out_ch["Cholesterol"] == 0])
assign_mean(df_out_ch[df_out_ch["Cholesterol"] != 0], not_df_out_ch, 'Cholesterol')

plt.figure(figsize=(20,10))
sns.histplot(data=df, x='Cholesterol', kde=True, palette=color, hue='HeartDisease')

# %% [markdown]
# #### Resting Blood Pressure

# %%
plt.figure(figsize=(20,5))
sns.boxplot(data=df, x="RestingBP")

# %%
not_df_out_rbp, df_out_rbp = detect_outliers('RestingBP')
print(f'Outliers in resting blood pressure represent the {round((df_out_rbp.shape[0]*100)/df.shape[0], 2)}% of our dataset')
df_out_rbp

# %% [markdown]
# In this case the outliers in resting blood pressure represent the 4.58% of total cases, we can delete them.

# %%
df = delete_outliers(df_out_rbp)
plt.figure(figsize=(20,10))
sns.histplot(data=df, x='RestingBP', kde=True, palette=color, hue='HeartDisease')

# %% [markdown]
# ### 📈 Others Visualizations:

# %%
px.scatter(data_frame=df, x="Age", y="MaxHR", color="HeartDisease")

# %%
px.scatter(data_frame=df, x="Oldpeak", y="MaxHR", color="HeartDisease")

# %%
px.scatter(data_frame=df, x="RestingBP", y="MaxHR", color="HeartDisease")

# %%
px.scatter(data_frame=df, x="Cholesterol", y="MaxHR", color="HeartDisease")

# %%
labels = ["Less chance of heart attack", "More chance of heart attack"]
values = [df[df["HeartDisease"] == 1].count().to_numpy()[0],
         df[df["HeartDisease"] == 0].count().to_numpy()[0]]

fig = go.Figure(data=[go.Pie(labels=labels, 
                             values=values, 
                             marker_colors=['cyan' ,'darkblue'],
                             textinfo='label+percent'
                            )])
fig.update(layout_title_text='Chance of heart disease', layout_showlegend=False)
fig.show()

# %%
labels = ["Female with less chance of HA",
          "Female with more chance of HA"]
values = [df[(df["Sex"] == 'F') & (df["HeartDisease"] == 0)].count().to_numpy()[0],
        df[(df["Sex"] == 'F') & (df["HeartDisease"] == 1)].count().to_numpy()[0]]

fig = go.Figure(data=[go.Pie(labels=labels, 
                             values=values,
                             marker_colors=['cyan' ,'darkblue'],
                             textinfo='label+percent'
                            )])
fig.update(layout_title_text='Probability of heart disease per woman', layout_showlegend=False)
fig.show()

# %%
labels = ["Male with less chance of HA",
          "Male with more chance of HA"]
values = [df[(df["Sex"] == 'M') & (df["HeartDisease"] == 0)].count().to_numpy()[0],
        df[(df["Sex"] == 'M') & (df["HeartDisease"] == 1)].count().to_numpy()[0]]

fig = go.Figure(data=[go.Pie(labels=labels, 
                             values=values,
                             marker_colors=['cyan' ,'darkblue'],
                             textinfo='label+percent'
                            )])
fig.update(layout_title_text='Probability of heart disease per man', layout_showlegend=False)
fig.show()

# %%
plt.figure(figsize=(20,40))
sns.pairplot(data=df[continuos_f + ["HeartDisease"]], hue="HeartDisease", palette='hls', kind="reg", corner=True, markers=["o", "s"], plot_kws={ 'scatter_kws': {'alpha': 0.8, 's':8}})

# %% [markdown]
# ****
# 
# # 🔢 Features encoding

# %% [markdown]
# ## Binary features enconding

# %%
df["Sex"] = df["Sex"].map({'M':1, 'F':0})
df["ExerciseAngina"] = df["ExerciseAngina"].map({'N':0, 'Y':1})
df

# %% [markdown]
# ## Data Binning

# %%
df_bin = df.copy()
for feature in continuos_f:
    bins = 5
    df_bin[feature] = pd.cut(df[feature], bins=bins, labels=range(bins)).astype(np.int64)
df_bin

# %% [markdown]
# ## One-hot encoding

# %%
df_dumm = pd.get_dummies(df_bin, columns = categorical_f+continuos_f)
df_dumm

# %% [markdown]
# ****
# 
# # ⚖️ Correlations between features and label:

# %%
df_corr = df_dumm.corr()
df_corr["HeartDisease"].sort_values(ascending=False)

# %%
# Scaling
#scaler = RobustScaler()

#df_clean = df_dumm.copy()
#df_clean[continuos_f] = scaler.fit_transform(df_clean[continuos_f].astype(np.float64))

#df_clean.head()

# %% [markdown]
# ****
# 
# # ✂️ Division into training and test sets

# %%
df_clean = df_dumm.copy()

X = df_clean.drop(["HeartDisease"], axis=1)
y = df_clean["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train

# %%
X_test

# %% [markdown]
# ****
# 
# # 🤖 Modeling

# %% [markdown]
# We will train three supervised learning models to tasks of classification also we will use grid search to tuning models' hyperparameters, additionally we will evaluate their performance with confusion matrix where:
# 
# <table>
#     <tr>
#         <th colspan="2" rowspan="2"></th>
#         <th colspan="2">Predicted</th>
#     </tr>
#     <tr>
#         <td>Negative</td>
#         <td>Positive</td>
#     </tr>
#     <tr>
#         <th rowspan="2">Actual</th>
#         <td>Negative</td>
#         <td>TN</td>
#         <td>FP</td>
#     </tr>
#     <tr>
#         <td>Positive</td>
#         <td>FN</td>
#         <td>TP</td>
#     </tr>
# </table>
# 
# Taking into account that:
# + Case negative: Patients with normal conditions (NC)
# + Case positive: Patients with CVDs
# + TN: The prediction tells us that the patient has NC when actually has NC.
# + TP: The prediction tells us that the patient has CVDs when actually has CVDs.
# + FN: The prediction tells us that the patient has NC when actually has CVDs.
# + FP: The prediction tells us that the patient has CVDs when actually has NC.
# 
# The worst case is a prediction of type FN, since we would be determining that the patient has normal condition, ignoring the possibility of CVD and exposing the patient to the risk of death. For this reason, we will focus to reduce these type of predictions.
# However, the amount of predictions of type FP shouldn't be too large since our model will be very useless.
# The score that help us to analysis the amount of predictions of type FN is the recall where:
# 
# $Recall = \frac{TP}{TP + FN}\quad\text{if}\quad FN \rightarrow 0 \Longrightarrow Recall \rightarrow 1$
# 
# Also:
# 
# $Precision = \frac{TP}{TP + FP}\quad\text{if}\quad FP \rightarrow 0 \Longrightarrow Precision \rightarrow 1$
# 
# and
# 
# $F1 = \frac{TP}{TP + \frac{FN + FP}{2}}\quad\text{if}\quad FN, FP \rightarrow 0 \Longrightarrow F1 \rightarrow 1$

# %%
metric = 'recall'

# %%
cv = 15
nFeatures = len(X_train.columns)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1]) 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Poisitive Rate")
    
def print_scores(y, y_pred):
    ac, pr, rc, f1 = accuracy_score(y, y_pred)*100 , precision_score(y, y_pred)*100, recall_score(y, y_pred)*100, f1_score(y, y_pred, average='weighted')*100
    print(f"Accuracy:{ac}")
    print(f"Precision:{pr}")
    print(f"Recall:{rc}")
    print(f"F1-score:{f1}")
    return {'ac': ac, 'pr':pr, 'rc':rc, 'f1':f1}

# %%
# For purpose of testing the before code.
#raise SystemExit()

# %% [markdown]
# ## 🟠 K-Nearest Neighbors Classifier

# %% [markdown]
# #### We train the model

# %%
param_grid = [{
    'n_neighbors':np.arange(5, 21),
    'weights':['uniform', 'distance'],
    'p':[1, 2],
}]

knn_clf = KNeighborsClassifier()
grid_knn = GridSearchCV(knn_clf, param_grid, cv=cv, scoring=metric)
grid_knn.fit(X_train, y_train)

# %%
best_knn_clf = grid_knn.best_estimator_
grid_knn.best_params_

# %%
y_train_pred1 = cross_val_predict(best_knn_clf, X_train, y_train, cv=cv)
conf_mx = confusion_matrix(y_train, y_train_pred1)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_train, y_train_pred1)
plot_roc_curve(fpr_knn, tpr_knn, "K-Nearest Neighbors")
plt.show()

# %%
roc_auc_score(y_train, y_train_pred1)

# %% [markdown]
# #### We evaluate the model with test set:

# %%
y_test_pred1 = cross_val_predict(best_knn_clf, X_test, y_test, cv=cv)
conf_mx = confusion_matrix(y_test, y_test_pred1)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_test_pred1)
plot_roc_curve(fpr_knn, tpr_knn, "K-Nearest Neighbors")
plt.show()

# %%
roc_auc_score(y_test, y_test_pred1)

# %% [markdown]
# #### We calculate the scores

# %%
knn_scores = print_scores(y_test, y_test_pred1)

# %% [markdown]
# ## 🟣 C-Support Vector Classifier.

# %% [markdown]
# #### We train the model

# %%
param_grid = [{
    "C": np.linspace(1, 1.5, 10),
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "gamma": ['scale', 'auto']
}]

svc_clf = SVC()
grid_svc = GridSearchCV(svc_clf, param_grid, scoring=metric, cv=cv)
grid_svc.fit(X_train, y_train)

# %%
best_svc_clf = grid_svc.best_estimator_
grid_svc.best_params_

# %%
y_train_pred2 = cross_val_predict(best_svc_clf, X_train, y_train, cv=cv)
conf_mx = confusion_matrix(y_train, y_train_pred2)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_train, y_train_pred2)
plot_roc_curve(fpr_svc, tpr_svc, "SVC")
plt.show()

# %%
roc_auc_score(y_train, y_train_pred2)

# %% [markdown]
# #### We evaluate the model with test set:

# %%
y_test_pred2 = cross_val_predict(best_svc_clf, X_test, y_test, cv=cv)
conf_mx = confusion_matrix(y_test, y_test_pred2)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_test_pred2)
plot_roc_curve(fpr_svc, tpr_svc, "SVC")
plt.show()

# %%
roc_auc_score(y_test, y_test_pred2)

# %% [markdown]
# #### We calculate the scores

# %%
svc_scores = print_scores(y_test, y_test_pred2)

# %% [markdown]
# ## 🟢 Random Forest Classifier

# %% [markdown]
# #### We train the model

# %%
nFeatures

# %%
param_grid = [{"n_estimators":[650, 700, 750, 800],
               'criterion':['gini','entropy'],
               "max_features":[nFeatures-9, nFeatures-8, nFeatures-5]
              }]
rf_clf = RandomForestClassifier(random_state=42)
grid_forest = GridSearchCV(rf_clf, param_grid, cv=cv, scoring=metric)
grid_forest.fit(X_train, y_train)

# %%
best_rf_clf = grid_forest.best_estimator_
best_rf_clf.get_params()

# %%
y_train_pred3 = cross_val_predict(best_rf_clf, X_train, y_train, cv=cv)
conf_mx = confusion_matrix(y_train, y_train_pred3)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_train_pred3)
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.show()

# %%
roc_auc_score(y_train, y_train_pred3)

# %% [markdown]
# #### We evaluate the model with test set:

# %%
y_test_pred3 = cross_val_predict(best_rf_clf, X_test, y_test, cv=cv)
conf_mx = confusion_matrix(y_test, y_test_pred3)
sns.heatmap(conf_mx, annot=True, fmt='')

# %%
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_test, y_test_pred3)
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.show()

# %%
roc_auc_score(y_test, y_test_pred3)

# %% [markdown]
# #### We calculate the scores

# %%
rf_scores = print_scores(y_test, y_test_pred3)

# %% [markdown]
# # 📏 Model Score Comparisons

# %%
scores = pd.DataFrame(data=[list(knn_scores.values()), list(svc_scores.values()), list(rf_scores.values())], columns=list(knn_scores.keys()))
scores = scores.transpose()
scores = scores.rename(columns={0:"K-Nearest Neighbors", 1:"C-Support Vector", 2:"Random Forest"})
scores.style.highlight_max(color = 'green', axis = 1)

# %% [markdown]
# According to our focus the best recall score was present for the Random Forest classifier, but also this classifier present the worst precision score, we must find a balance between recall score and precision score, <span style="color:green; font-weight: bold;">for me the best model is the KNN classifier.</span>

# %%
scores = pd.DataFrame({'Models':['KNN','SVC','RF'],'ACC':[roc_auc_score(y_test, y_test_pred1),roc_auc_score(y_test, y_test_pred2),roc_auc_score(y_test, y_test_pred3)]})

# %%
scores

# %%
sns.barplot(data = scores, x = 'Models', y = 'ACC')

# %%
x = df_clean.drop(["HeartDisease"], axis=1)
y = df_clean["HeartDisease"]

# %%
knn = KNeighborsClassifier()
knn.fit(x,y)

# %%
#Exporting model using joblib library
import joblib
joblib.dump(knn_clf,"hdp_model.pkl")



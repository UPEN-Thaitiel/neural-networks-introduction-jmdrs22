# %% [markdown]
# ## Heart Failure

# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.set(style="whitegrid")

# %% [markdown]
# ### Import Dataset

# %%
df = pd.read_csv("heart.csv")
df.head()

# %% [markdown]
# ### Data Inspection

# %%
df.shape

# %%
df.dtypes

# %%
df.isnull().sum()

# %%
df.describe()

# %% [markdown]
# ### Exploratory Data Analysis

# %% [markdown]
# #### 1. Target variable distribution: This bar chart shows how many patients do and do not have heart disease. It's useful for determining whether the data set is balanced between the two classes.

# %%
sns.countplot(data=df, x='HeartDisease')
plt.title("Distribución de la variable objetivo")
plt.show()

# %% [markdown]
# #### 2. Age distribution: This graph shows the age distribution in the dataset. The KDE (density estimate) curve shows how the most common patient ages cluster.

# %%
sns.histplot(df['Age'], kde=True)
plt.title("Distribución de Edad")
plt.show()

# %% [markdown]
# #### 3. Boxplot for cholesterol (Outlier detection): This boxplot allows you to identify cholesterol distribution and potential outliers. Points outside the whiskers represent outliers that could affect the analysis.

# %%
sns.boxplot(x='Cholesterol', data=df)
plt.title("Outliers en colesterol")
plt.show()

# %% [markdown]
# #### 4. Violin Plot: This graph shows the distribution of cholesterol levels according to the presence or absence of heart disease. It combines information from a boxplot with the density of the data, facilitating comparisons between the two classes.

# %%
sns.violinplot(data=df, x="HeartDisease", y="Cholesterol")
plt.title("Cholesterol Distribution by Heart Disease")
plt.show()


# %% [markdown]
# #### 5.KDE Plot (Kernel Density Estimation): This graph compares the distribution of maximum heart rate (MaxHR) between patients with and without heart disease. The curves allow for smooth and continuous visualization of differences in trends within each group.

# %%
sns.kdeplot(data=df[df['HeartDisease'] == 0]['MaxHR'], label="No Disease", fill=True)
sns.kdeplot(data=df[df['HeartDisease'] == 1]['MaxHR'], label="Disease", fill=True)
plt.title("MaxHR Distribution by Heart Disease")
plt.legend()
plt.show()


# %% [markdown]
# #### 6. Pair Plot: This graph shows the relationships between pairs of numerical variables, differentiated by heart disease class. It is useful for identifying patterns, correlations, and possible class separations.

# %%
selected = ['Age', 'MaxHR', 'Oldpeak', 'Cholesterol', 'HeartDisease']
sns.pairplot(df[selected], hue='HeartDisease')
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# %% [markdown]
# ##### This scatter plot shows the relationship between age and maximum heart rate, differentiating by the presence of disease. It is useful for detecting possible clusters, trends, or separations between classes.

# %%
sns.scatterplot(data=df, x='Age', y='MaxHR', hue='HeartDisease')
plt.title("Age vs MaxHR by Heart Disease")
plt.show() 

# %% [markdown]
# ### 7. Function for bivariate analysis (histogram + KDE): This function generates two side-by-side graphs for a variable: a histogram and a density curve (KDE), both separated by heart disease class. It allows you to visualize and compare how that variable is distributed between patients with and without disease.

# %%
def bivariable_analysis(var, var_title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)

    # Histogram by HeartDisease
    sns.histplot(data=df[df['HeartDisease'] == 0], x=var, label='No Disease', ax=ax[0], color='skyblue')
    sns.histplot(data=df[df['HeartDisease'] == 1], x=var, label='Disease', ax=ax[0], color='salmon')
    ax[0].set_xlabel(var_title)
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Histogram by Heart Disease')
    ax[0].legend()

    # KDE Plot by HeartDisease
    sns.kdeplot(data=df[df['HeartDisease'] == 0], x=var, label='No Disease', fill=True, ax=ax[1], color='skyblue')
    sns.kdeplot(data=df[df['HeartDisease'] == 1], x=var, label='Disease', fill=True, ax=ax[1], color='salmon')
    ax[1].set_xlabel(var_title)
    ax[1].set_title('Density Plot by Heart Disease')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

bivariable_analysis('Age', 'Age')
bivariable_analysis('MaxHR', 'Maximum Heart Rate')
bivariable_analysis('Oldpeak', 'Oldpeak (ST depression)')
bivariable_analysis('Cholesterol', 'Cholesterol')




# %% [markdown]
# ### 8. Boxplot de RestingBP: This graph shows the distribution of resting blood pressure (RBP) by heart disease class. It allows for comparing medians and ranges and detecting potential outliers between the two groups.

# %%
sns.boxplot(data=df, x='HeartDisease', y='RestingBP')
plt.title("RestingBP por clase")
plt.show()

# %% [markdown]
# ### Extra. 3D scatter plot: This three-dimensional graph visualizes the joint relationship between age, maximum heart rate, and peak age. Color indicates the presence or absence of heart disease, making it easier to observe patterns or groupings among classes.

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Age'], df['MaxHR'], df['Oldpeak'], c=df['HeartDisease'], cmap='coolwarm', alpha=0.7)

ax.set_xlabel('Age')
ax.set_ylabel('MaxHR')
ax.set_zlabel('Oldpeak')
ax.set_title('3D: Age vs MaxHR vs Oldpeak')
plt.show()


# %% [markdown]
# ### Feature Transformation

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# get_dummies()
# 
# What it does Why it's needed? get_dummies() Encodes text into numbers
# Why is it necessary? Because models don't understand text

# %%
df_encoded = pd.get_dummies(df, drop_first=True)

# %% [markdown]
# separate x and y
# 
# What it does Why it's needed? Define predictors and objective
# Why is it necessary? To train models correctly

# %%
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# %% [markdown]
# StandardScaler()
# 
# What it does Why it's needed? Normalize the data
# Why is it necessary? To improve model performance

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ### Modeling 

# %% [markdown]
# Model 1: Logistic Regression

# %% [markdown]
# We import the logistic regression model and the metrics to evaluate how well the model performs.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %% [markdown]
# We split the data into training (80%) and testing (20%) to train the model and then check how it performs.

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %% [markdown]
# We create the logistic regression model and train it using the training data.

# %%
lr = LogisticRegression()
lr.fit(X_train, y_train)

# %% [markdown]
# We use the trained model to make predictions on the test data.

# %%
y_pred_lr = lr.predict(X_test)

# %% [markdown]
# We store the model results in a dictionary to see how well it predicts: accuracy, precision, recall, F1 score, and the confusion matrix.

# %%
lr_results = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1 Score': f1_score(y_test, y_pred_lr),
    'Confusion Matrix': confusion_matrix(y_test, y_pred_lr)
}

lr_results

# %%
from sklearn.metrics import ConfusionMatrixDisplay

# %% [markdown]
# This block visually displays the confusion matrix for the logistic regression model. It's useful because it shows how often the model correctly or incorrectly predicted whether a patient has heart disease or not.
# We include it to better visualize the model’s performance, instead of just looking at the raw numbers.

# %%
ConfusionMatrixDisplay(confusion_matrix=lr_results["Confusion Matrix"],display_labels=["No Disease", "Disease"]).plot(cmap="YlGnBu")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# %% [markdown]
# Model 2: Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# %%
y_pred_dt = dt.predict(X_test)

# %%
dt_results = {
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "Precision": precision_score(y_test, y_pred_dt),
    "Recall": recall_score(y_test, y_pred_dt),
    "F1 Score": f1_score(y_test, y_pred_dt),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_dt)
}

dt_results

# %% [markdown]
# This block displays the confusion matrix for the decision tree model. It helps us see how many predictions were correct and how many were wrong, separating patients with and without heart disease.
# We include it because it’s a visual tool that makes it easier to analyze the model’s performance.

# %%
ConfusionMatrixDisplay(confusion_matrix=dt_results["Confusion Matrix"],display_labels=["No Disease", "Disease"]).plot(cmap="YlGnBu")
plt.title("Confusion Matrix: Decision Tree")
plt.show()


# %% [markdown]
# Model 3: Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# %%
y_pred_rf = rf.predict(X_test)

# %%
rf_results = {
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1 Score": f1_score(y_test, y_pred_rf),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_rf)
}

rf_results

# %% [markdown]
# This block shows the confusion matrix for the Random Forest model. It allows us to see how the model classified patients with and without heart disease, showing correct and incorrect predictions.
# We include it because it’s a clear visual way to evaluate model performance.

# %%
ConfusionMatrixDisplay(confusion_matrix=rf_results["Confusion Matrix"],display_labels=["No Disease", "Disease"]).plot(cmap="YlGnBu")
plt.title("Confusion Matrix: Random Forest")
plt.show()


# %% [markdown]
# We create a DataFrame called results_df that contains key results (precision, accuracy, etc.) from the three models: Logistic Regression, Decision Tree, and Random Forest.
# Then, we drop the confusion matrix column to display only the most relevant metrics in a cleaner, comparative table.

# %%
results_df = pd.DataFrame([lr_results, dt_results, rf_results])
results_df.drop("Confusion Matrix", axis=1)


# %% [markdown]
# ### Task 3: Classification Use Case with scikit-learn MLPClassifier

# %% [markdown]
# Import and configure MLPClassifier + GridSearchCV

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# %%
df = pd.read_csv("heart.csv")
df.head()

# %%
df_encoded = pd.get_dummies(df, drop_first=True)


# %%
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# %%
# Definir modelo base
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Definir el grid de hiperparámetros
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (10,10)],
    'alpha': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh']
}

# Configurar GridSearchCV
grid_search = GridSearchCV(
    mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
)

# Entrenar
grid_search.fit(X_train, y_train)



# %%
# %%
best_mlp = grid_search.best_estimator_
y_pred_mlp = best_mlp.predict(X_test)


# %%
# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mlp_results = {
    "Model": "MLPClassifier",
    "Accuracy": accuracy_score(y_test, y_pred_mlp),
    "Precision": precision_score(y_test, y_pred_mlp),
    "Recall": recall_score(y_test, y_pred_mlp),
    "F1 Score": f1_score(y_test, y_pred_mlp),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_mlp)
}

mlp_results

# %%
# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(confusion_matrix=mlp_results["Confusion Matrix"],display_labels=["No Disease", "Disease"]).plot(cmap="YlGnBu")
plt.title("Confusion Matrix: MLPClassifier")
plt.show()


# %%
# %%
results_df = pd.concat([results_df, pd.DataFrame([mlp_results])], ignore_index=True)
results_df.drop("Confusion Matrix", axis=1)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

names = ["Nearest Neighbors","Nearest Neighbours2", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3)
    KNeighborsClassifier(n_neighbors=3,weights='distance'),
    SVC(kernel="linear", C=0.025),
    SVC(C=50,kernel='rbf',gamma=1)
    DecisionTreeClassifier(max_depth=13)
    RandomForestClassifier(max_depth=13, max_features=2)
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=13),n_estimators=100, learning_rate=.02)
    GaussianNB()
    LDA(),
    QDA()]

datasets = [[Y,Class]]

i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    noise_len =1000000
    noise = np.random.uniform(-50,70,(noise_len,2))
    X_train = np.vstack([X_train,noise])
    y_train = np.append(y_train,np.linspace(13,13,noise_len))
    i += 1

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        print('Training:',name,score)
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('Testing:',name,score)
        i += 1
      


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

X = np.linspace(-2, 2, 7)
y = X ** 3 # original dependecy 

plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');

#%%

# You'll need to build a tree with only one node (also called root) that contains all train observations (instances). 
# How will predictions of this tree look like for $x \in [-2, 2]$? 
# Create an appropriate plot using a pen, paper and Python if needed (but no sklearn is needed yet).

x_1 = [i for i in X if i<0]
x_1 = np.asarray(x_1)
y_1 = x_1 **3

x_2 = [i for i in X if i>=0]
x_2 = np.asarray(x_2)
y_2 = x_2 **3


figure = plt.figure(figsize=(20,15))

plt.subplot2grid((2,2),(0,0))
plt.scatter(X, y)
plt.title("row data")

plt.subplot2grid((2,2),(1,0))
plt.scatter(x_1, y_1, color = 'red')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title("x<0 split")

plt.subplot2grid((2,2),(1,1))
plt.scatter(x_2, y_2, color = 'blue')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title("x>=0 split")
plt.show()

#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

data = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/mlbootcamp5_train.csv", sep=";", index_col='id')
#data = pd.read_csv("/Users/akif/Desktop/Data Science/python/Kodluyoruz/assignment2-classification-maozsoy/mlbootcamp5_train.csv", index_col='id')

data["age"] = data["age"] / 365.25

# data["cholesterol_1"] = ["1" if i==1 else "0" for i in data.cholesterol]
# data["cholesterol_2"] = ["1" if i==2 else "0" for i in data.cholesterol]
# data["cholesterol_3"] = ["1" if i==3 else "0" for i in data.cholesterol]

cholesterol = pd.get_dummies(data["cholesterol"], prefix = "cholesterol", dummy_na=False)
gluc = pd.get_dummies(data["gluc"], prefix = "gluc", dummy_na=False)
data = pd.concat([data, cholesterol, gluc], axis=1)
data.drop(["cholesterol", "gluc"], axis=1, inplace=True)


y = data.cardio.values
x = data.drop(["cardio"], axis=1).values
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.30, random_state=17)


dt = DecisionTreeClassifier(max_depth=3, random_state=17)
dt.fit(x_train, y_train)

# yöntem_1
column_names = list(data.drop(["cardio"], axis=1).columns)
dot_data = tree.export_graphviz(dt, out_file='dot_data_decision_tree.dot', feature_names=[u'{}'.format(c) for c in column_names], class_names= [str(x) for x in dt.classes_], rounded=True, special_characters=True)

# yöntem_2_ÇALIŞMADI
#import pydotplus
#from sklearn.tree import export_graphviz
#def tree_graph_to_png(tree, feature_names, png_file_to_save):
#    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
#    graph = pydotplus.graph_from_dot_data(tree_str)  
#    graph.write_png(png_file_to_save)
#tree_graph_to_png(tree=decision_tree, feature_names=column_names, png_file_to_save='C:\Users\004202\Desktop\2_ANALİTİK\Pyhton_anlatım\kişisel çalışma\ders4_dt_yontem2.png')


y_pred = dt.predict(x_valid)

dt_acc_score_train = round(dt.score(x_train, y_train) * 100, 2) 
print("Accuracy Score(Train) of DT(no grid):", dt_acc_score_train)

dt_acc_score_valid = round(dt.score(x_valid, y_valid) * 100, 2) 
print("Accuracy Score(Validation) of DT(no grid):", dt_acc_score_valid)


param_grid = {'max_depth': np.arange(1, 11)}
dt_grid = DecisionTreeClassifier(random_state=17)
dt_grid_cv = GridSearchCV(dt_grid, param_grid, cv=5) 
dt_grid_cv.fit(x_train, y_train)
print("max depth:", dt_grid_cv.best_params_)
print("best score", dt_grid_cv.best_score_)


dt_grid = DecisionTreeClassifier(max_depth=dt_grid_cv.best_params_['max_depth'])
dt_grid.fit(x_train, y_train)
y_pred_grid = dt_grid.predict(x_train)

dt_g_acc_score_train = round(dt_grid.score(x_train, y_train) * 100, 2) 
print("Accuracy Score(Train) of DT(grid):", dt_g_acc_score_train)

dt_g_acc_score_valid = round(dt_grid.score(x_valid, y_valid) * 100, 2) 
print("Accuracy Score(Validation) of DT(grid):", dt_g_acc_score_valid)



accur_train = []
accur_valid = []
max_depth_ = []
for i in range(1,11):
    dt2 = DecisionTreeClassifier(max_depth=i, random_state=17)
    dt2.fit(x_train, y_train)
    dt2_acc_score_train = round(dt2.score(x_train, y_train) * 100, 2)
    dt2_acc_score_valid = round(dt2.score(x_valid, y_valid) * 100, 2)
    accur_train.append(dt2_acc_score_train)
    accur_valid.append(dt2_acc_score_valid)
    max_depth_.append(i)

plt.plot(max_depth_, accur_train, color= "red", label= "Train Data")
plt.plot(max_depth_, accur_valid, color= "green", label= "Validation Data")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

acc_change_train = ((dt_g_acc_score_train - dt_acc_score_train) / dt_acc_score_train) * 100
acc_change_valid = ((dt_g_acc_score_valid - dt_acc_score_valid) / dt_acc_score_valid) * 100

print("Increasing ppt of Accuracy(train data): {}".format(acc_change_train))
print("Increasing ppt of Accuracy(valid data): {}".format(acc_change_valid))

data["age_40_50"] = [1 if 40 <= i < 50 else 0 for i in data.age]
data["age_50_55"] = [1 if 50 <= i < 55 else 0 for i in data.age]
data["age_55_60"] = [1 if 55 <= i < 60 else 0 for i in data.age]
data["age_60_65"] = [1 if 60 <= i < 65 else 0 for i in data.age]

data["systolic_bp_120_140"] = [1 if 120 <= i < 140 else 0 for i in data.ap_hi]
data["systolic_bp_140_160"] = [1 if 140 <= i < 160 else 0 for i in data.ap_hi]
data["systolic_bp_160_180"] = [1 if 160 <= i < 180 else 0 for i in data.ap_hi]



male = pd.get_dummies(data["gender"], drop_first=True, prefix = "male")
data = pd.concat([data, male], axis=1)
data.drop(["gender", "age", "ap_hi"], axis=1, inplace=True)


column_names2 = list(data.drop(["cardio"], axis=1).columns)
for i in column_names2:
    print(i, data.i.unique())





































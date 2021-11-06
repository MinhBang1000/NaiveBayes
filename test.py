import pandas as pd
dt = pd.read_csv("Training Data.csv",delimiter=',')
dt_test = pd.read_csv("Test Data.csv",delimiter=',')
# Có 252000 phần tử dùng để train
# Có 11 thuộc tính và 1 nhãn
# Nhãn chính là Risk_Flag có 2 loại nhãn 0 và 1
# Thuộc tính Income  10.3k đến 10.00m
# Thuộc tính Age 21 đến 79
# Thuộc tính Experience 0 đến 20
# Thuộc tính Married/Single 2 giá trị married hoặc Single
# Thuộc tính sỡ hữu nhà ở là rented hoặc norent_noown hoặc owned
# Thuộc tính Car_owned Yes hoặc No
# Thuộc tính Professional gồm 
# Physician
# Statistician
# Web_designer
# Psychologist
# Computer_hardware_engineer
# Thuộc tính CITY 317 giá trị khác nhau
# Thuộc tính STATE 
# Thuộc tính CURRENT HOURS

# Tiền xử lý
X_train = dt.iloc[:,1:12] 
Y_train = dt.Risk_Flag
X_test = dt_test.iloc[:,1:12]
Y_test = pd.read_csv("Sample Prediction Dataset.csv",delimiter=',').iloc[:,1]

# Dự đoán bằng Cây quyết định
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
dtree = DecisionTreeClassifier(criterion="gini",random_state=100)
dtree.fit(X_train,Y_train)
Y_predict = dtree.predict(X_test)
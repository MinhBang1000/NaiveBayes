import pandas as pd
import numpy as np
dt = pd.read_csv("play_tennis.csv")
print(dt)
X = dt.iloc[:, 0:4]
Y = dt.Play
for i in range(0, len(X.iloc[0, :])):
    unique_arr = X.iloc[:, i].unique()
    print(unique_arr)  # Đây là các giá trị trong mỗi cột thuộc tính

# Cách hình thành mảng 2 chiều có trọng số là tên thuộc tính và nhãn
nhan = dt.Play.unique()
Outlook = dt.Outlook.unique()
temp = np.array([[2, 3], [4, 0], [3, 2]])
dtf = pd.DataFrame(temp, index=Outlook, columns=nhan)

# Cách chuyển từ Pandas về Numpy
a1 = np.array([[2, 3], [3, 4]])
a2 = np.array([[2, 3], [3, 4], [5, 6]])
a = np.array([])

# Dictionary
thisdict1 = {
    "test1": "1",
    "test2": "2"
}
thisdict = {
    "brand": thisdict1,
    "model": a2,
    "year": 1964
}
# Khởi tạo
nhan = dt.Play
thuoctinh = dt.Outlook
giatrinhan = dt.Play.unique()
giatrithuoctinh = thuoctinh.unique()
tmp = {}
dic_tt = {}
for k in range(0, len(giatrithuoctinh)):
    dic_tt[giatrithuoctinh[k]] = {}
    tmp = {}
    for h in range(0, len(giatrinhan)):
        tmp[giatrinhan[h]] = 0
    dic_tt[giatrithuoctinh[k]] = tmp

for i in range(0, len(giatrithuoctinh)):  # Phải có dấu xuống dòng tại hai dòng for riêng biệt
    for j in range(0, len(thuoctinh)):
        if (thuoctinh[j] == giatrithuoctinh[i]):
            dic_tt[thuoctinh[j]][nhan[j]] += 1

# Lấy ra từng cột
for i in range(0, len(dt.columns)):
    print(dt.to_numpy()[:, i])
dt.columns[0]


class MyNavieBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X_train, Y_train):
        nhan = Y_train
        dic_total = {}
        for tt_index in range(0, len(X_train.columns)):
            thuoctinh = X_train.to_numpy()[:, tt_index]
            giatrinhan = nhan.unique()
            giatrithuoctinh = thuoctinh.unique()
            tmp = {}
            dic_tt = {}
            for k in range(0, len(giatrithuoctinh)):
                dic_tt[giatrithuoctinh[k]] = {}
                tmp = {}
                for h in range(0, len(giatrinhan)):
                    tmp[giatrinhan[h]] = 0
                dic_tt[giatrithuoctinh[k]] = tmp

            # Phải có dấu xuống dòng tại hai dòng for riêng biệt
            for i in range(0, len(giatrithuoctinh)):
                for j in range(0, len(thuoctinh)):
                    if (thuoctinh[j] == giatrithuoctinh[i]):
                        dic_tt[thuoctinh[j]][nhan[j]] += 1

            dic_total[X_train.columns[tt_index]] = dic_tt

from NaiveBayes import MyNavieBayes
import pandas as pd
import numpy as np
dt = pd.read_csv("play_tennis.csv")
print(dt)
X = dt.iloc[:, 0:4]
Y = dt.Play
nv = MyNavieBayes()
nv.fit(X_train=X,Y_train=Y)
nv.predict(X)

X.iloc[:,3]
temp = X.iloc[0,:]


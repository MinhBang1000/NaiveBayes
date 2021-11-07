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
    "test1": [1,2,3,4],
    "test2": {
        "a":1,
        "b2":2
    }
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
#=========================================================================================
from NaiveBayes import MyNavieBayes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dt = pd.read_csv("play_tennis.csv")
print(dt)
X = dt.iloc[:, 0:4]
Y = dt.Play
X_train = X.iloc[0:11,:]
Y_train = Y.iloc[0:11]
X_test = X.iloc[11:,:]
Y_test = Y.iloc[11:]
# Tách tập dữ liệu Train và Test
nv = MyNavieBayes()
nv.fit(X_train=X_train,Y_train=Y_train,Laplace=1)
Y_pre = nv.predict(X_test)
print(accuracy_score(Y_test,Y_pre)*100,"%")
#=========================================================================================
dt_pr = pd.read_csv("Training Data.csv",delimiter=",").iloc[:,1:13]
X_train = dt_pr.iloc[0:84000,0:11]
Y_train = dt_pr.Risk_Flag.iloc[0:84000]
dt_pr_te = dt_pr                        # Chia 1/3 tập dữ liệu đầu tiên để Test
X_test = dt_pr_te.iloc[84000:,0:11]
Y_test = dt_pr_te.Risk_Flag.iloc[84000:]
# Cách phân biệt liên tục hay không
# Rời rạc thì là ít nhất 1 chữ và có ít hơn 10 giá trị khác nhau unique
# Liên tục là phần còn lại
# Dùng hàm check_number_or_string để check xem là số hay chữ
# Dùng len(unique để check xem là rời rạc hay liên tục)

def check_number_or_string(val):
    try:
        test = int(val)
        print("Val is number")
    except:
        try:
            test = float(val)
            print("Val is number")
        except:
            print("Val is string")

def check_continue(col): # Đưa vào một thuộc tính, một cột
    check = False # Là dạng liên tục, True là rời rạc
    for i in range(0,len(col)):
        check_1 = True
        try:
            test = int(col[i])
            check_1 = False
        except:
            try:
                test = float(col[i])
                check_1 = False
            except:
                check_1 = True
        if (check_1 == True): # Nếu có tìm thấy một chuỗi thì đây là dạng rời rạc
            check = check_1
            break # Đi đến điều kiện số 2 để xem đây có phải rời rạc hay không
        
    if (len(col.unique())<=10 and check==False): # Chỉ khi nó không có cái nào là chữ và có ít hơn 10 giá trị khác nhau thì nó là kiểu rời rạc
        check = True
    return check

check_continue(X_train.iloc[:,0])
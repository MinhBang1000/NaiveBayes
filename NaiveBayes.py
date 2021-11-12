import math

import numpy as np


class MyNavieBayes:
    def __init__(self) -> None:
        pass
    
    def check_continue(self,col): # Đưa vào một thuộc tính, một cột
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
            
        if (len(np.unique(col.flatten()))<=10 and check==False): # Chỉ khi nó không có cái nào là chữ và có ít hơn 10 giá trị khác nhau thì nó là kiểu rời rạc
            check = True
        return check

    def fit(self, X_train, Y_train, Laplace):
        nhan = Y_train.values.flatten()
        dic_total = {}
        dic_nhan = {}
        for n_index in range(0,len(np.unique(Y_train.values.flatten()))): # Khởi tạo từ điển nhãn
            dic_nhan[np.unique(Y_train.values.flatten())[n_index]] = 0
        for m_index in range(0,len(Y_train)):          # Cộng dồn số lượng từng nhãn vào từ điển 
            dic_nhan[Y_train.values.flatten()[m_index]]+=1
        for tt_index in range(0, len(X_train.columns)):# Duyệt qua từng thuộc tính
            thuoctinh = X_train.iloc[:, tt_index].values.flatten()      # Lấy một thuộc tính 
            giatrinhan = np.unique(Y_train.values.flatten())
            giatrithuoctinh = np.unique(X_train.iloc[:, tt_index].values.flatten()) # Lấy các giá trị phân biệt của thuộc tính
            tmp = {}
            tmp1 = {}
            dic_tt = {}
            dic_tmp1 = {}
            if (self.check_continue(thuoctinh)==True): #Check xem là rời rạc hay liên tục
                for k in range(0, len(giatrithuoctinh)):    # Cộng dồn số lượng các giá trị của thuộc tính theo nhãn vào từ điển con
                    dic_tt[giatrithuoctinh[k]] = {}
                    dic_tmp1[giatrithuoctinh[k]] = {}
                    tmp = {}
                    tmp1 = {}
                    for h in range(0, len(giatrinhan)):
                        tmp[giatrinhan[h]] = 0
                        tmp1[giatrinhan[h]] = 0
                    dic_tt[giatrithuoctinh[k]] = tmp
                    dic_tmp1[giatrithuoctinh[k]] = tmp1

                # Phải có dấu xuống dòng tại hai dòng for riêng biệt
                for i in range(0, len(giatrithuoctinh)):
                    for j in range(0, len(thuoctinh)):
                        if (thuoctinh[j] == giatrithuoctinh[i]):
                            dic_tt[thuoctinh[j]][nhan[j]] += 1
                            dic_tmp1[thuoctinh[j]][nhan[j]] += 1
                
                for j in range(0,len(giatrinhan)): # Nếu biết có một thuộc tính trong một giá trị bằng 0 thì sẽ quay lại thuộc tính đầu tiên của giá trị nhãn đó để tính lại theo Laplace
                    check_zero = False # Để check giá trị thuộc tính bằng 0
                    check_back = False
                    i=0
                    while (i<len(giatrithuoctinh)):
                        if (dic_tt[giatrithuoctinh[i]][giatrinhan[j]]!=0 and check_zero==False):
                            dic_tt[giatrithuoctinh[i]][giatrinhan[j]] = dic_tt[giatrithuoctinh[i]][giatrinhan[j]]/dic_nhan[giatrinhan[j]]
                        else:
                            check_zero = True # Để làm Laplace cho tất cả các giá trị của thuộc tính đó
                            if (check_back==False):
                                i = 0 # Quay lại cái đầu nếu đang ở các giá trị sau của thuộc tính này
                                check_back = True # Chỉ set i bằng 0 một lần
                        if (check_zero==True): # Xử lý Laplace
                            # print("Gia tri thuoc tinh = ",giatrithuoctinh[i])
                            # print("Dem duoc = ",dic_tmp1[giatrithuoctinh[i]][giatrinhan[j]])
                            # print("Bao nhieu nhan nay = ",1/(len(dic_tt)))
                            # print("Laplace = ",Laplace)
                            dic_tt[giatrithuoctinh[i]][giatrinhan[j]] = (dic_tmp1[giatrithuoctinh[i]][giatrinhan[j]] + (1/(len(dic_tt)))*Laplace)/(dic_nhan[giatrinhan[j]]+Laplace)
                        i+=1
            else: # Kiểu liên tục
                X_lt = thuoctinh # Một cột thuộc tính
                Y_lt = giatrinhan
                import math
                for kiemtra_lt in range(0,len(giatrinhan)):
                    tong_theo_nhan = 0
                    so_luong = 0
                    for lt_tt in range(0,len(X_lt)):
                        if (giatrinhan[kiemtra_lt]==Y_train.values.flatten()[lt_tt]): # Đếm số lượng để tính
                            tong_theo_nhan = tong_theo_nhan + X_lt[lt_tt]
                            so_luong = so_luong + 1
                    u = tong_theo_nhan/so_luong
                    o2 = 0
                    for lt in range(0,len(X_lt)):
                        if (giatrinhan[kiemtra_lt]==Y_train.values.flatten()[lt]):
                            o2 = o2+(X_lt[lt]-u)*(X_lt[lt]-u)
                    o2 = o2*(1/(so_luong-1))
                    o = math.sqrt(o2)
                    tmp[giatrinhan[kiemtra_lt]] = {
                        "u":u,
                        "o2":o2,
                        "o":o
                    }
                    # print("Gia tri nhan: ",giatrinhan[kiemtra_lt])
                dic_tt["X"] = tmp # Thêm X để đồng bộ với từ điển ở trên

            dic_total[X_train.columns[tt_index]] = dic_tt
        
        self.dic_main = dic_total
        for i in range(0,len(giatrinhan)):
            dic_nhan[giatrinhan[i]] = dic_nhan[giatrinhan[i]]/Y_train.count() 
        self.dic_label = dic_nhan
        self.total_label = Y_train.count()
        self.giatrinhan = np.unique(Y_train.values.flatten())
    
    def predict(self, X_test):
        import math
        ret_arr = []
        for k in range(0,len(X_test)): # Duyệt qua từng dòng
            X = X_test.iloc[k,:] # Chỉ cần thay đổi chỗ này để cho nhiều dòng
            max_xacsuat = -1
            ten_nhan = ''
            for i in range(0,len(self.giatrinhan)): # Kiểm tra là Yes hay No trên từng dòng
                xacsuat = self.dic_label[self.giatrinhan[i]]
                for j in range(0,len(X)):
                    check_lien_tuc = False # True la thuoc tinh lien tuc
                    for key in self.dic_main[X_test.columns[j]].keys():
                        if (key=="X"):
                            check_lien_tuc = True
                            break
                    if (check_lien_tuc==True): # Kiểu dữ liệu cuả thuộc tính là liên tục
                        tmp_dic = self.dic_main[X_test.columns[j]]["X"][self.giatrinhan[i]] # Từ điển này lưu u, o2, o
                        gx = -(X.iloc[j]-tmp_dic["u"])*(X.iloc[j]-tmp_dic["u"])
                        ex = 2*tmp_dic["o2"]
                        f = (1/math.sqrt(2*math.pi)*tmp_dic["o"])*math.pow(math.e,gx/ex)
                        xacsuat = xacsuat*f
                    else:
                        xacsuat = xacsuat*self.dic_main[X_test.columns[j]][X.iloc[j]][self.giatrinhan[i]]
                if (xacsuat>=max_xacsuat).all(): # lựa chọn nhãn
                    max_xacsuat = xacsuat
                    ten_nhan = self.giatrinhan[i]
            ret_arr.append(ten_nhan)
        return ret_arr
    
    def show(self):
        print(self.dic_main)
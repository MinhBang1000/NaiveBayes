class MyNavieBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X_train, Y_train):
        nhan = Y_train
        dic_total = {}
        dic_nhan = {}
        for n_index in range(0,len(Y_train.unique())): # Khởi tạo từ điển nhãn
            dic_nhan[Y_train.unique()[n_index]] = 0
        for m_index in range(0,len(Y_train)):          # Cộng dồn số lượng từng nhãn vào từ điển 
            dic_nhan[Y_train[m_index]]+=1
        for tt_index in range(0, len(X_train.columns)):# Duyệt qua từng thuộc tính
            thuoctinh = X_train.iloc[:, tt_index]      # Lấy một thuộc tính 
            giatrinhan = nhan.unique()
            giatrithuoctinh = X_train.iloc[:, tt_index].unique() # Lấy các giá trị phân biệt của nhãn và thuộc tính
            tmp = {}
            dic_tt = {}
            for k in range(0, len(giatrithuoctinh)):    # Cộng dồn số lượng các giá trị của thuộc tính theo nhãn vào từ điển con
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

            for i in range(0,len(giatrithuoctinh)):
                for j in range(0,len(giatrinhan)):
                    dic_tt[giatrithuoctinh[i]][giatrinhan[j]] = dic_tt[giatrithuoctinh[i]][giatrinhan[j]]/dic_nhan[giatrinhan[j]]

            dic_total[X_train.columns[tt_index]] = dic_tt
        
        self.dic_main = dic_total
        for i in range(0,len(giatrinhan)):
            dic_nhan[giatrinhan[i]] = dic_nhan[giatrinhan[i]]/Y_train.count() 
        self.dic_label = dic_nhan
        self.total_label = Y_train.count()
        self.giatrinhan = Y_train.unique()
    
    def predict(self, X_test):
        X = X_test.iloc[2,:] # Chỉ cần thay đổi chỗ này để cho nhiều dòng
        max_xacsuat = -1
        ten_nhan = ''
        for i in range(0,len(self.giatrinhan)): # Kiểm tra là Yes hay No trên từng dòng
            xacsuat = self.dic_label[self.giatrinhan[i]]
            for j in range(0,len(X)):
                xacsuat = xacsuat*self.dic_main[X_test.columns[j]][X.iloc[j]][self.giatrinhan[i]]
            if (xacsuat>=max_xacsuat):
                max_xacsuat = xacsuat
                ten_nhan = self.giatrinhan[i]
        print(ten_nhan)
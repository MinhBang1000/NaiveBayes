class MyNavieBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X_train, Y_train):
        nhan = Y_train
        dic_total = {}
        dic_nhan = {}
        for n_index in range(0,len(Y_train.unique())):
            dic_nhan[Y_train.unique()[n_index]] = 0
        for m_index in range(0,len(Y_train)):
            dic_nhan[Y_train[m_index]]+=1
        for tt_index in range(0, len(X_train.columns)):
            thuoctinh = X_train.iloc[:, tt_index]
            giatrinhan = nhan.unique()
            giatrithuoctinh = X_train.iloc[:, tt_index].unique()
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
        
        self.dic_main = dic_total
        self.dic_label = dic_nhan
    
    def predict(self):
        print(self.dic_main)
        print(self.dic_label)
        
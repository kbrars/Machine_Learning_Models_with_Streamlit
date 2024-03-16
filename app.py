import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import streamlit.components.v1 as components
import streamlit as st


class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
        self.control ="None"
    def run(self):
        self.get_dataset()
        self.pre_proccessing_add()
        self.get_classifier_add()
        self.confusion_matrix_add()

    #ekrana yazdırılacak kısım
    def Init_Streamlit_Page(self):
        st.title("Makine Öğrenmesi Model Değerlendirmesi")


        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer', 'Yeni Veri Seti Ekle ')
        )
        st.write(f"## {self.dataset_name}")
     
        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )

    def dataset_info(self):
        st.write("Veri seti boyutu:  ",self.data.shape)
        st.write("Sınıf Sayısı  ",len(np.unique(self.data['diagnosis'])))
        
    #dataset yükleme
    def get_dataset(self):
        if self.dataset_name == "Breast Cancer":
            self.data = pd.read_csv("datasets/data.csv")
            self.dataset_info()
            st.write("Veri seti ilk 10 satır:")
            st.write(self.data.head(10))
        else:
            uploaded_file = st.file_uploader("Advertising.csv", type=["csv"])  #csv dosyası yükleme 
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                self.dataset_info()
                st.write("Veri seti ilk 10 satır:")
                st.write(self.data.head(10))
            else:
                self.control =False #veri seti yüklenmediyse
                
    #ön işeleme  
    def pre_proccessing(self):
        st.header('Ön İşleme ', divider='rainbow')
        st.markdown("- Veri setinden gereksiz olan 'id' ve 'Unnamed: 32' sütunları silindi. ")
        st.markdown("- Diagnosis sütununda bulunan 'M'  değeri 1, 'B'  değeri ise 0 ile sayısallaştırıldı.")
        st.markdown("- Veri setindeki özellikler X, etiket y olarak ikiye ayrıldı.")
        st.markdown("- Özelliklerin değerleri normalize edildi.")

        self.data.drop(['id','Unnamed: 32'], axis = 1, inplace= True )
        self.data["diagnosis"] = [1 if diag == "M" else 0 for diag in self.data["diagnosis"]]
        st.write("Veri Seti son 10 satırı")
        st.write(self.data.tail(10))

        M = self.data[self.data["diagnosis"] == 1]
        B = self.data[self.data["diagnosis"] == 0]

        st.write("Korelasyon matrisi")
        fig = plt.figure(figsize=(16,8))
        plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu",alpha= 0.3)
        plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Iyi",alpha= 0.3)
        plt.xlabel("radius_mean")
        plt.ylabel("texture_mean")
        plt.legend()
        st.pyplot(plt)

        self.y = self.data["diagnosis"].values
        self.X = self.data.drop(["diagnosis"] ,axis = 1)


        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        st.write("")
        st.write("Normalize edilmiş özelliklerin(X) değerleri: ")
        st.write(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size = 0.2, random_state = 32)
    
    def pre_proccessing_add(self):
        if self.dataset_name != "Breast Cancer":
            if self.control: #veri seti yüklendiyse diğer tüm ön işleme aşamalarını yap.
                self.pre_proccessing()
        else:
            self.pre_proccessing()


    #model
    def train_model_with_gridsearch(self,param_grid,model):
        grid = GridSearchCV(model, param_grid, cv=10)
        
        grid = grid.fit(self.X_train, self.y_train)
        self.best_params=grid.best_params_
        return grid
    
    def evulate_model(self,model):
        st.header('Model Eğitimi', divider='rainbow')
        st.markdown(f"- Classifier: {self.classifier_name}")
        st.markdown(f"- En iyi parametreler:  {self.best_params}")
        self.y_pred = model.predict(self.X_test)
        self.accuarcy = model.score(self.X_test,self.y_test)
        st.success(f'Accuarcy: {self.accuarcy}')
        
    def get_classifier_and_train(self):
        if self.classifier_name == 'KNN':
            #train
            knn = KNeighborsClassifier()
            param_grid = {"n_neighbors": np.arange(1,30,1)}
            knn = self.train_model_with_gridsearch(param_grid, knn)
            #evulate
            self.evulate_model(knn)

        elif self.classifier_name == 'SVM':
            #train
            svm = SVC()
            param_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
            svm= self.train_model_with_gridsearch(param_grid, svm)
            #evulate
            self.evulate_model(svm)

        else:
            #train
            gNB = GaussianNB()
            param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
            gNB= self.train_model_with_gridsearch(param_grid, gNB)
            #evulate
            self.evulate_model(gNB)


    def get_classifier_add(self):
        if self.dataset_name != "Breast Cancer":
            if self.control: #veri seti yüklendiyse diğer tüm ön işleme aşamalarını yap.
                self.get_classifier_and_train()
        else:
            self.get_classifier_and_train()

    def confusion_matrix(self):
        st.write("Confusion Matrix")
        cm = confusion_matrix(self.y_test, self.y_pred) 
        f, ax = plt.subplots(figsize = (5,5))

        sns.heatmap(cm,annot = True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        st.pyplot(plt)

    def confusion_matrix_add(self):
        if self.dataset_name != "Breast Cancer":
            if self.control: #veri seti yüklendiyse diğer tüm ön işleme aşamalarını yap.
                self.confusion_matrix()
        else:
            self.confusion_matrix()
            
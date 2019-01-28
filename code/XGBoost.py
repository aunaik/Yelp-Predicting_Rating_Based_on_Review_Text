# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:41:18 2018

@author: jigar
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn import decomposition

class XGBRatingPredictor():
    def __init__(self,filename,skipRow,xStart,xEnd,yColumn,model):
        #read data
        self.dataset = loadtxt(filename, delimiter=",", skiprows=skipRow)
        
        # split data into X and y
        self.X = self.dataset[:,xStart:xEnd]
        self.Y = self.dataset[:,yColumn]
        
        self.modelUsed = model
        if model == 'XGBoost':
            #Set XGBoost Classifier
            self.model = XGBClassifier()
        
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None        
        self.predictions = None
        
    def setXGBparameters(self,delta_step = None,alpha = None):
        if delta_step != None:
            self.model.max_delta_step = 1
        if alpha != None:
            self.model.reg_alpha = 1
        self.model.objective = 'multi:softmax'
        self.model.max_depth = 7
        self.model.booster = 'gbtree'
            
    def normalizeData(self,normalizer,axis = 1):
        if normalizer == 'Quantile':
            self.X = QuantileTransformer(n_quantiles=10, random_state=0).fit_transform(self.X)
        elif normalizer == 'Robust':
            self.X = RobustScaler(quantile_range=(25, 75)).fit_transform(self.X)
        else:
            self.X = normalize(self.X,axis = axis)
    
    def pcaData(self,components):
        pca = decomposition.PCA(n_components=components)
        pca.fit(self.X)
        self.X = pca.transform(self.X)
    
    def oversampleData(self,method):
        if method == 1:
            smote = SMOTEENN(random_state=0)
        else:
            smote = SMOTETomek(random_state=0)
        self.X_train, self.Y_train = smote.fit_sample(self.X_train, self.Y_train)  
        
    def splitData(self,testSize):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = testSize, random_state = 7) 
    
    def predictSamples(self):       
        if self.modelUsed == 'XGBoost':
            self.model.fit(self.X_train, self.Y_train)
            y_pred = self.model.predict(self.X_test)
        self.predictions = [round(value) for value in y_pred]
        
    def evaluations(self,accuracy=False,meanAbsError=False,showConfusionMatrix=False):
        
        if accuracy == True:
            accuracy = accuracy_score(self.Y_test, self.predictions)
            print("Accuracy : "+str(accuracy * 100.0))
        
        sumOfError = 0.0
        samplingError = {}
        for i in range(5):
            samplingError[i] = {}
            for j in range(5):
                samplingError[i][j] = 0
        
        for i in range(len(self.Y_test)):
            
            error = (abs(self.Y_test[i] - self.predictions[i]))
            sumOfError += error
            
            if showConfusionMatrix == True:
                samplingError[int(self.Y_test[i])-1][int(self.predictions[i])-1] += 1
                
        print('\nMean Absolute Error : '+str(sumOfError/len(self.Y_test)))
        
        print('Confusion Matrix')
        print(' \t1\t2\t3\t4\t5\tCorrect Prediction %')
        for i in range(0,5):
            text = str(i+1)
            count = 0
            for j in range(0,5):
                text += '\t'+str(samplingError[i][j])
                count += samplingError[i][j]
            text += '\t'+str((samplingError[i][i]/count)*100)
            print(text)
                           
def main():
    
    predictor = XGBRatingPredictor('reviewTable150b.csv',skipRow=1,xStart=0,xEnd=601,yColumn=601,model='XGBoost')
    predictor.normalizeData('Robust')    
    predictor.pcaData(50)
    predictor.splitData(0.1)
    predictor.oversampleData(2)
    predictor.predictSamples()
    predictor.evaluations(True,True,True)
    
if __name__ == "__main__":
    main()
     



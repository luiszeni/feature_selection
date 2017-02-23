import numpy as np
import math
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

# this class stores and plot the classification results
class ClassificationResults:
    def __init__(self, x):   
        self.numberFeatures = x.shape[1]
        self.acc_f_rf = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_knn = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_svm_linear = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_svm_poly = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_svm_rbf = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_svm_sigmoid = np.zeros(( self.numberFeatures+1), dtype=float)
        self.acc_f_nn = np.zeros(( self.numberFeatures+1), dtype=float)
        self.selectedFeatures = np.zeros(( self.numberFeatures+1), dtype='|S128')

    
    def plotData(self):
   
        plt.figure()
        plt.plot(self.acc_f_rf,label='Random Forest')
        plt.plot(self.acc_f_knn,label='Knn')
        plt.plot(self.acc_f_svm_linear,label='Svm kernel linear')
        plt.plot(self.acc_f_svm_poly,label='Svm kernel poly')
        plt.plot(self.acc_f_svm_rbf,label='Svm kernel rbf')
        plt.plot(self.acc_f_svm_sigmoid,label='Svm kernel sigmoid')
        plt.plot(self.acc_f_nn,label='Neural Network')
        
        plt.ylabel('accurancy')
        plt.xlabel('number of top features')
        plt.xlim([1, self.numberFeatures-1])
        plt.ylim([0.3,1.0])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)
                       
               
    def plotPNG(self,location):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.acc_f_rf,label='Random Forest')
        ax.plot(self.acc_f_knn,label='Knn')
        ax.plot(self.acc_f_svm_linear,label='Svm kernel linear')
        ax.plot(self.acc_f_svm_poly,label='Svm kernel poly')
        ax.plot(self.acc_f_svm_rbf,label='Svm kernel rbf')
        ax.plot(self.acc_f_svm_sigmoid,label='Svm kernel sigmoid')
        ax.plot(self.acc_f_nn,label='Neural Network')
        
        ax.set_xlabel('number of top features')
        ax.set_ylabel('accurancy')

        ax.set_xlim([1, self.numberFeatures])
        ax.set_ylim([0.3,1.0])

        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        ax.grid('on')
        fig.savefig(location, bbox_extra_artists=(lgd,), bbox_inches='tight')

class FeatureSelection:
    
    def __init__(self, x, y, method):
        self.numberSamples = x.shape[0]
        self.numberFeatures = x.shape[1]
        self.method = method
        self.x = x; 
        self.y = y;
        
        if self.method  == 'cross_randomforest': 
            print "cross_randomforest"
            self.featuresImp = np.zeros((self.numberSamples,self.numberFeatures), dtype=float)
        
            for i in range(0, self.numberSamples):
              
                x_train = np.concatenate((x[0:i,:], x[i+1:,:]),axis=0);
                y_train = np.concatenate((y[0:i], y[i+1:]),axis=0);
                
                forest = RandomForestClassifier(30);       
                forest.fit(x_train, y_train)    
        
                feat =  forest.feature_importances_ 
                self.featuresImp[i] = np.argsort(feat)[::-1]
        elif self.method  == 'cross_logisticregression': 
            print "cross_logisticregression"
 
        elif self.method  == 'randomforest':  
            print "randomforest"
            forest = RandomForestClassifier(30);       
            forest.fit(x, y)    
            feat =  forest.feature_importances_ 
            self.featuresImp = np.argsort(feat)[::-1]
        elif self.method  == 'logisticregression': 
            print "logisticregression"
        else:
            print "invalid method"
        
    def getNTopFeatures(self, nTFeat):
        
        if self.method == 'cross_randomforest': 
            top = self.featuresImp[:,0:nTFeat];
            feat = np.zeros((self.numberFeatures), dtype=float)
            
            for i in range(0, self.numberSamples):
                for j in range(0, nTFeat):
                    idx = int(top[i,j])
                    feat[idx] =  feat[idx]+1
                    
                
            top = np.argsort(feat)[::-1];
            top = top[0:nTFeat]        
            return top
        elif self.method  == 'cross_logisticregression':
            self.featuresImp = np.zeros((self.numberSamples,nTFeat), dtype=float)
        
            for i in range(0, self.numberSamples):
              
                x_train = np.concatenate((self.x[0:i,:], self.x[i+1:,:]),axis=0);
                y_train = np.concatenate((self.y[0:i], self.y[i+1:]),axis=0);
                
                model = LogisticRegression()
                rfe = RFE(model, nTFeat)
                rfe = rfe.fit(x_train, y_train)
                feat = rfe.support_;
                
                feat = np.array(np.where(feat)[0])                
                
                self.featuresImp[i] = feat
                #self.featuresImp[i] = np.argsort(feat)[::-1]
            
            top = self.featuresImp;
            feat = np.zeros((self.numberFeatures), dtype=float)
            
            for i in range(0, self.numberSamples):
                for j in range(0, nTFeat):
                    idx = int(top[i,j])
                    feat[idx] =  feat[idx]+1
                    
                
            top = np.argsort(feat)[::-1];
            top = top[0:nTFeat]        
            return top
            
        elif self.method  == 'randomforest' :
            top = self.featuresImp[0:nTFeat] 
            return top
        elif self.method  == 'logisticregression':  
            model = LogisticRegression()
            rfe = RFE(model, nTFeat)
            rfe = rfe.fit(self.x, self.y)
            feat = rfe.support_;
            return np.array(np.where(feat)[0])
        else:
           return -1;
       
    
    
    

def removeNaFromData(data, verbose = True):
    #Cleaning NaN data
    idxToDelete = []
    for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]): 
                if  math.isnan(data[i,j]):
                    #print "Line " + str(i) + " " + str(db2Data[i, :])
                    idxToDelete.append(i);
                    break;
            
    data = np.delete(data, idxToDelete, axis=0)  
    if verbose:    
        print "Removed " + str(len(idxToDelete)) + " lines in database 2"
    return data;
    
def loadDatabase(fileUrl, removeFeaturesNumber, clearNa = True ):
    featureNames =  genfromtxt(fileUrl, delimiter=';', dtype='str')[0,:]
    x =  genfromtxt(fileUrl, delimiter=';')[1:,:]

    x = removeNaFromData(x)

    x = np.delete(x, removeFeaturesNumber, axis=1)  
    featureNames = np.delete(featureNames, removeFeaturesNumber, axis=0)
    y = x[:,1]
    x = x[:,2:]
    
    featureNames = featureNames[2:] 
    
    return x,y,featureNames


    plt.show()

def testDataOnClassifyers(xTrain, yTrain, xTest, yTest, results, numbFeat, verbose = True):
    #SVM 
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(xTrain, yTrain)    
    results.acc_f_svm_linear[numbFeat] = svm_linear.score(xTest,yTest)
    
    svm_poly = SVC(kernel='poly')
    svm_poly.fit(xTrain, yTrain)    
    results.acc_f_svm_poly[numbFeat] = svm_poly.score(xTest,yTest);
   
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(xTrain, yTrain)    
    results.acc_f_svm_rbf[numbFeat] = svm_rbf.score(xTest,yTest)
       
    svm_sigmoid = SVC(kernel='sigmoid')
    svm_sigmoid.fit(xTrain, yTrain)    
    results.acc_f_svm_sigmoid[numbFeat] = svm_sigmoid.score(xTest,yTest)
       
    #forest
    forest = RandomForestClassifier(30);
    forest.fit(xTrain, yTrain)    
    results.acc_f_rf[numbFeat] = forest.score(xTest,yTest)
   
    #knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xTrain, yTrain) 
    results.acc_f_knn[numbFeat] = knn.score(xTest,yTest)

    #nn
    nn =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                       hidden_layer_sizes=(5, 2), random_state=1)
    nn.fit(xTrain, yTrain) 
    results.acc_f_nn[numbFeat] = nn.score(xTest,yTest)
    
    if verbose:
         print "svm_linear " + str(results.acc_f_svm_linear[numbFeat]);   
         print "svm_poly " +  str(results.acc_f_svm_poly[numbFeat]);
         print "svm_rbf " +  str(results.acc_f_svm_rbf[numbFeat]);
         print "sigmoid " +  str(results.acc_f_svm_sigmoid[numbFeat]);
         print "forest " +  str(results.acc_f_rf[numbFeat]);
         print "knn " +  str(results.acc_f_knn[numbFeat]);
         print "NN " +  str(results.acc_f_nn[numbFeat]);
    return



def leaveOneOut(xDb,yDb,topFeatures,nF,results):
    x = xDb[:,topFeatures[:nF]]
    y = yDb
    

    numberSamples = x.shape[0]
    
    acc_rf = np.zeros((numberSamples), dtype=float)
    acc_knn = np.zeros((numberSamples), dtype=float)
    acc_svm_linear = np.zeros((numberSamples), dtype=float)
    acc_svm_poly = np.zeros((numberSamples), dtype=float)
    acc_svm_rbf = np.zeros((numberSamples), dtype=float)
    acc_svm_sigmoid = np.zeros((numberSamples), dtype=float)    
    acc_nn = np.zeros((numberSamples), dtype=float)    
    
    
    
    for i in range(0, numberSamples):
        x_train = np.concatenate((x[0:i,:], x[i+1:,:]),axis=0);
        y_train = np.concatenate((y[0:i], y[i+1:]),axis=0);
        x_test = x[i,:];  
        y_test = y[i];  
       
        #Rando Forests       
        forest = RandomForestClassifier(30);
        forest.fit(x_train, y_train)    
        acc_rf[i] = forest.predict(x_test.reshape(1, -1))
        if acc_rf[i] == y_test:
            acc_rf[i] = 1
        else:
            acc_rf[i]= 0
            
        #knn
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)    
        acc_knn[i] = knn.predict(x_test.reshape(1, -1))
        if acc_knn[i] == y_test:
            acc_knn[i] = 1
        else:
            acc_knn[i]= 0
            
    
        #svm  linear
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(x_train, y_train)    
        acc_svm_linear[i] = svm_linear.predict(x_test.reshape(1, -1))
        
        if acc_svm_linear[i] == y_test:
            acc_svm_linear[i] = 1
        else:
            acc_svm_linear[i]= 0
        
        #svm  poly
        svm_poly = SVC(kernel='poly')
        svm_poly.fit(x_train, y_train)    
        acc_svm_poly[i] = svm_poly.predict(x_test.reshape(1, -1))
        
        if acc_svm_poly[i] == y_test:
            acc_svm_poly[i] = 1
        else:
            acc_svm_poly[i]= 0
         
        #svm  rbf
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(x_train, y_train)    
        acc_svm_rbf[i] = svm_rbf.predict(x_test.reshape(1, -1))
        
        if acc_svm_rbf[i] == y_test:
            acc_svm_rbf[i] = 1
        else:
            acc_svm_rbf[i]= 0
            
        #svm  sigmoid
        svm_sigmoid = SVC(kernel='sigmoid')
        svm_sigmoid.fit(x_train, y_train)    
        acc_svm_sigmoid[i] = svm_sigmoid.predict(x_test.reshape(1, -1))
        
        if acc_svm_sigmoid[i] == y_test:
            acc_svm_sigmoid[i] = 1
        else:
            acc_svm_sigmoid[i]= 0

        ##NN
        nn =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        nn.fit(x_train, y_train)    
        acc_nn[i] = nn.predict(x_test.reshape(1, -1))
        
        if acc_nn[i] == y_test:
            acc_nn[i] = 1
        else:
            acc_nn[i]= 0    
    
    correctCRf =  float(acc_rf[acc_rf == 1].shape[0]);
    correctPRf = (correctCRf)/float(numberSamples);
    results.acc_f_rf[nF] = correctPRf
   
    correctCknn =   float(acc_knn[acc_knn == 1].shape[0]);
    correctPknn = (correctCknn)/ float(numberSamples);
    results.acc_f_knn[nF] = correctPknn
    
    correctCsvm_linear =   float(acc_svm_linear[acc_svm_linear == 1].shape[0]);
    correctPsvm_linear = (correctCsvm_linear)/ float(numberSamples);
    results.acc_f_svm_linear[nF] = correctPsvm_linear
    
    correctCsvm_poly =   float(acc_svm_poly[acc_svm_poly == 1].shape[0]);
    correctPsvm_poly = (correctCsvm_poly)/ float(numberSamples);
    results.acc_f_svm_poly[nF] = correctPsvm_poly
    
    correctCsvm_rbf =   float(acc_svm_rbf[acc_svm_rbf == 1].shape[0]);
    correctPsvm_rbf = (correctCsvm_rbf)/ float(numberSamples);
    results.acc_f_svm_rbf[nF] = correctPsvm_rbf
    
    correctCsvm_sigmoid =   float(acc_svm_sigmoid[acc_svm_sigmoid == 1].shape[0]);
    correctPsvm_sigmoid = (correctCsvm_sigmoid)/ float(numberSamples);
    results.acc_f_svm_sigmoid[nF] = correctPsvm_sigmoid
    
    correctCnn =   float(acc_nn[acc_nn == 1].shape[0]);
    correctPnn = (correctCnn)/ float(numberSamples);
    results.acc_f_nn[nF] = correctPnn
    
    print "results using only " + str(nF)+ " top features in RandomForest: " + str(correctPRf)
    print "results using only " + str(nF)+ " top features in knn: " + str(correctPknn)   
    print "results using only " + str(nF)+ " top features in svm linear: " + str(correctPsvm_linear)
    print "results using only " + str(nF)+ " top features in svm poly: " + str(correctPsvm_poly)
    print "results using only " + str(nF)+ " top features in svm rbf: " + str(correctPsvm_rbf)
    print "results using only " + str(nF)+ " top features in svm sigmoid: " + str(correctPsvm_sigmoid)
    print "results using only " + str(nF)+ " top features in NN: " + str(correctPnn)

    return
    


def createResultsMatrix(results):
    
    matrix = np.zeros((8, results.acc_f_knn.shape[0]), dtype='|S128')
    
    matrix[0,0] = 'Features/Classifiers'
    matrix[0,1:] = results.selectedFeatures[1:]
    matrix[1,0] = 'KNN'
    matrix[1,1:] = results.acc_f_knn.astype('|S6')[1:]
    
    matrix[2,0] = 'NN'
    matrix[2,1:] = results.acc_f_nn.astype('|S6')[1:]    
    
    
    matrix[3,0] = 'R. Forest'
    matrix[3,1:] = results.acc_f_rf.astype('|S6')[1:]    
    
    matrix[4,0] = 'Svm Linear'
    matrix[4,1:] = results.acc_f_svm_linear.astype('|S6')[1:]    
    
    matrix[5,0] = 'Svm Poly'
    matrix[5,1:] = results.acc_f_svm_poly.astype('|S6')[1:]    
    
    matrix[6,0] = 'Svm Rbf'
    matrix[6,1:] = results.acc_f_svm_rbf.astype('|S6')[1:] 
    
    matrix[7,0] = 'Svm Sigmoid'
    matrix[7,1:] = results.acc_f_svm_sigmoid.astype('|S6')[1:]    
    
    
    return matrix
    

def exportResults(name, results):
    pd.DataFrame(createResultsMatrix(results)).to_csv("results/" + name + ".csv")
    results.plotPNG("results/" + name + ".png");

def runAllFeatureSelectors(x, y, expName, featureNames):    

    
    cfs =  FeatureSelection(x,y, "cross_logisticregression")
    
    results = ClassificationResults(x)
    numberFeatures = x.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        leaveOneOut(x, y ,top,nF,results)
    
    exportResults(expName + "_cross_logisticregression", results)
    
    cfs =  FeatureSelection(x,y, "cross_randomforest")
    
    results = ClassificationResults(x)
    numberFeatures = x.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        leaveOneOut(x, y,top,nF,results)
    
    exportResults(expName + "_cross_randomforest", results)    

    cfs =  FeatureSelection(x,y, "logisticregression")
    
    results = ClassificationResults(x)
    numberFeatures = x.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        leaveOneOut(x, y,top,nF,results)
    
    exportResults(expName + "_logisticregression", results)
    
    
    
    
    cfs =  FeatureSelection(x,y, "randomforest")
    
    results = ClassificationResults(x)
    numberFeatures = x.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        leaveOneOut(x, y, top,nF,results)
    
    exportResults(expName + "_randomforest", results)    

    
def runAllFeatureSelectorsTwoDatasets(xTrain,xVal, yTrain, yVal, expName, featureNames):    

    
    cfs =  FeatureSelection(xTrain,yTrain, "cross_logisticregression")
    
    results = ClassificationResults(xVal)
    numberFeatures = xTrain.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        testDataOnClassifyers(xTrain[:,top],yTrain,xVal[:,top],yVal,results, nF)  
        #leaveOneOut(xDb2,yDb2,top,nF,results)
    
    exportResults(expName + "_cross_logisticregression", results)
    
    cfs =  FeatureSelection(xTrain,yTrain, "cross_randomforest")
    
    results = ClassificationResults(xVal)
    numberFeatures = xTrain.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        testDataOnClassifyers(xTrain[:,top],yTrain,xVal[:,top],yVal,results, nF)  
        #leaveOneOut(xDb2,yDb2,top,nF,results)
    
    exportResults(expName + "_cross_randomforest", results)


    cfs =  FeatureSelection(xTrain,yTrain, "logisticregression")
    
    results = ClassificationResults(xVal)
    numberFeatures = xTrain.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        testDataOnClassifyers(xTrain[:,top],yTrain,xVal[:,top],yVal,results, nF)  
        #leaveOneOut(xDb2,yDb2,top,nF,results)
    
    exportResults(expName + "_logisticregression", results)


    cfs =  FeatureSelection(xTrain,yTrain, "randomforest")
    
    results = ClassificationResults(xVal)
    numberFeatures = xTrain.shape[1]
    
    for nF in range (1,numberFeatures + 1):   
        top = cfs.getNTopFeatures(nF);
        results.selectedFeatures[nF] = ', '.join(featureNames[top])
        testDataOnClassifyers(xTrain[:,top],yTrain,xVal[:,top],yVal,results, nF)  
        #leaveOneOut(xDb2,yDb2,top,nF,results)
    
    exportResults(expName + "_randomforest", results)    
    
    


def main():
	
    #Experiment 1 - avaliate the datasets individually unsing all features
    xDb2, yDb2,featureNames2 =  loadDatabase('banco2.csv',[])
    standard_scaler = StandardScaler()
    xDb2 = standard_scaler.fit_transform(xDb2)

    runAllFeatureSelectors(xDb2, yDb2, "ex1/Base2_todos_atributos", featureNames2)
    

    xDb3, yDb3,featureNames3 =  loadDatabase('banco3.csv',[])
    standard_scaler = StandardScaler()
    xDb3 = standard_scaler.fit_transform(xDb3)
    
    runAllFeatureSelectors(xDb3, yDb3, "ex1/Base3_todos_atributos", featureNames3)



    #Experiment 2 - avaliate the datasets individually unsing only common features
    xDb2, yDb2,featureNames2 =  loadDatabase('banco2.csv',[10,11,12])
    standard_scaler = StandardScaler()
    xDb2 = standard_scaler.fit_transform(xDb2)

    runAllFeatureSelectors(xDb2, yDb2, "ex2/Base2_atributos_comuns", featureNames2)
    

    xDb3, yDb3,featureNames3 =  loadDatabase('banco3.csv',[10])
    standard_scaler = StandardScaler()
    xDb3 = standard_scaler.fit_transform(xDb3)
    
    runAllFeatureSelectors(xDb3, yDb3, "ex2/Base3_atributos_comuns", featureNames3)

    #Experiment 3 - evaluate the datasets together
    xDb2, yDb2,featureNames2 =  loadDatabase('banco2.csv',[10,11,12])
    xDb3, yDb3,featureNames3 =  loadDatabase('banco3.csv',[10])
    
    xAll = np.concatenate((xDb2,xDb3),axis=0);
    yAll = np.concatenate((yDb2,yDb3),axis=0);
    
    standard_scaler = StandardScaler()
    xAll = standard_scaler.fit_transform(xAll)
   
    runAllFeatureSelectors(xAll, yAll, "ex3/Base2_atributos_comuns", featureNames2)
    
    #Experiment 4 - one dataset to train, one to test.
    xDb2, yDb2,featureNames2 =  loadDatabase('banco2.csv',[10,11,12])
    xDb3, yDb3,featureNames3 =  loadDatabase('banco3.csv',[10])

    standard_scaler = StandardScaler()
    xDb2 = standard_scaler.fit_transform(xDb2)
    xDb3 = standard_scaler.transform(xDb3) #dataset 3 is not used in the normalization

    runAllFeatureSelectorsTwoDatasets(xDb2, xDb3, yDb2, yDb3, "ex4/Base2_treino_vs_Base3_teste", featureNames2)

    xDb2, yDb2,featureNames2 =  loadDatabase('banco2.csv',[10,11,12])
    xDb3, yDb3,featureNames3 =  loadDatabase('banco3.csv',[10])

    standard_scaler = StandardScaler()
    xDb3 = standard_scaler.fit_transform(xDb3)
    xDb2 = standard_scaler.transform(xDb2) #dataset 2 is not used in the normalization

    runAllFeatureSelectorsTwoDatasets(xDb3, xDb2, yDb3, yDb2, "ex4/Base3_treino_vs_Base2_teste", featureNames2)


if __name__ == "__main__":
    main()
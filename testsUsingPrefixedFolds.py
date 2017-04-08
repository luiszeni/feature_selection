from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

import numpy as np






def processFold(name, fold, data, gtData):


        
    numberOfFolds = int(fold.max(1).max(0))
    numerOfMethods= 7;
    numberOfIterations = 50
    numberOfRepetitionsForEach = 100
    numeberOfFeatures = data.shape[1]

    results = np.zeros((numeberOfFeatures, numerOfMethods, numberOfIterations, numberOfFolds, numberOfRepetitionsForEach ), dtype=float)
    selectedFeatures = np.zeros((numeberOfFeatures, numerOfMethods, numberOfIterations, numberOfFolds,numeberOfFeatures), dtype=bool)


    cont = 0
    total = numeberOfFeatures * numberOfIterations * numberOfFolds * numberOfRepetitionsForEach
    for nF in range(0,numeberOfFeatures):
        for i in range(0,numberOfIterations):
        
            for f in range(0,numberOfFolds):
                
                # subtract one to work with ranges between [0 N-1] to make easy to work with numpy indexes        
                iterationFolds = np.subtract(fold[:,i], 1)
                         
                #extract data for each folds
                x_train = data[iterationFolds != f]
                y_train = gtData[iterationFolds != f]
                
                ##top nF fetaures extraction
                
                model = LogisticRegression()
                rfe = RFE(model, nF + 1) # as index nF starts on zero add one to start in one
                rfe = rfe.fit(x_train, y_train)
                topFeatures = rfe.support_;
                
                #log features for future usage :) i am proud of this solution, :P
                selectedFeatures[nF,0,i,f,:] = topFeatures         
                            

                selecetedData = data[:, topFeatures]            
                
                x_train = selecetedData[iterationFolds != f]
                ##test data
                x_test = selecetedData[iterationFolds == f]
                y_test = gtData[iterationFolds == f]
                print cont,"of", total
                for r in range(0,numberOfRepetitionsForEach):         
                    cont = cont + 1
                            
                    
                    svm_linear = SVC(kernel='linear')
                    svm_linear.fit(x_train, y_train)    
                    results[nF,0,i,f,r] = svm_linear.score(x_test,y_test)
                    
                                
                    svm_poly = SVC(kernel='poly')
                    svm_poly.fit(x_train, y_train)     
                    results[nF,1,i,f,r] =  svm_poly.score(x_test,y_test)
                   
                    svm_rbf = SVC(kernel='rbf')
                    svm_rbf.fit(x_train, y_train)   
                    results[nF,2,i,f,r] =  svm_rbf.score(x_test,y_test)
                       
                    svm_sigmoid = SVC(kernel='sigmoid')
                    svm_sigmoid.fit(x_train, y_train)    
                    results[nF,3,i,f,r] =  svm_sigmoid.score(x_test,y_test)
                       
                    #forest
                    forest = RandomForestClassifier(30);
                    forest.fit(x_train, y_train)       
                    results[nF,4,i,f,r] =  forest.score(x_test,y_test)
                   
                    #knn
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(x_train, y_train)   
                    results[nF,5,i,f,r] =  knn.score(x_test,y_test)
                
                    #nn
                    nn =  MLPClassifier(solver='lbfgs', alpha=1e-5,
                                       hidden_layer_sizes=(5, 2), random_state=1)
                    nn.fit(x_train, y_train)   
                    results[nF,6,i,f,r] =  nn.score(x_test,y_test)
        
    np.save("results/"+name+"results.nparray", results) 
    np.save("results/"+name+"selectedFeatures.nparray", selectedFeatures)



def main():
    #Load the prefixed data from the folds
    foldsDataFolder = 'foldsClaudio/'

    normalizatedData = foldsDataFolder + '/dados.txt'
    groudTruth = foldsDataFolder +'/gt.txt'

    data = np.loadtxt(normalizatedData, dtype='float')
    gtData = np.loadtxt(groudTruth, dtype='float')



    '''
    _2Folds50It = foldsDataFolder + '/2folds_50iterations.txt'
    _2FoldData = np.loadtxt(_2Folds50It, dtype='float')
    processFold("2folds_50iterations", _2FoldData, data, gtData)
   
    _5Folds50It = foldsDataFolder + '/5folds_50iterations.txt'
    _5FoldData = np.loadtxt(_5Folds50It, dtype='float')
    processFold("5folds_50iterations", _5FoldData, data, gtData)
    '''
    _10Folds50It = foldsDataFolder + '/10folds_50iterations.txt'
    _10FoldData = np.loadtxt(_10Folds50It, dtype='float')
    processFold("10folds_50iterations", _10FoldData, data, gtData)


   


    
if __name__ == "__main__":
    main()







import numpy as np
import numpy as np
import matplotlib.pyplot as plt

#Load the prefixed data from the folds
results = np.load('results/10folds_50iterationsresults.nparray.npy')
labels = np.load('results/10folds_50iterationsselectedFeatures.nparray.npy')

foldsName = "10 folds 50 iterations"

numberOfFolds = results.shape[3]
numerOfMethods= results.shape[1];
numberOfIterations = results.shape[2]
numberOfRepetitionsForEach = results.shape[4]
numeberOfFeatures = results.shape[0]

'''
mean = np.zeros((numeberOfFeatures, numerOfMethods, numberOfIterations ), dtype=float)
std = np.zeros((numeberOfFeatures, numerOfMethods, numberOfIterations ), dtype=float)

for nF in range(0,numeberOfFeatures):
    plt.figure()    
    for m in range (0, numerOfMethods):  
        for i in range(0,numberOfIterations):

               
            mean[nF,m,i] =  results[nF,m,i,:,:].mean()
            std[nF,m,i] = results[nF,m,i,:,:].std()
            
    
        plt.errorbar(range(0,50), mean[nF,m,:],  std[nF,m,:], fmt='*')
        #plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
    plt.savefig(str(nF) + ".jpg")
#print results.shape
'''
mean = np.zeros((numeberOfFeatures, numerOfMethods ), dtype=float)
std = np.zeros((numeberOfFeatures, numerOfMethods ), dtype=float)

    
for nF in range(0,numeberOfFeatures):

    for m in range (0, numerOfMethods):  
     
        mean[nF,m] =  results[nF,m,:,:,:].mean()
        std[nF,m] = results[nF,m,:,:,:].std().mean()
            
    valores = ['svm linear', 'svm poly', 'svm rbf', 'svm sigmoid', 'random forest', 'knn', 'neural net']
    
for nF in range(0,numerOfMethods):
    plt.figure()       
    
    plt.errorbar(range(1,numeberOfFeatures+1), mean[:,nF],  std[:,nF], fmt='*')
    plt.title(foldsName + " -  " + valores[nF])
    plt.xlim([0,12])
    plt.xlabel("number of N top features")
    
    plt.savefig(foldsName + " -  " + valores[nF] + ".jpg")
#print results.shape
#print labels.shape
            
          
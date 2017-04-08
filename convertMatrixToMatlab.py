import numpy as np, scipy.io


results2 = np.load('results/2folds_50iterationsresults.nparray.npy')
labels2 = np.load('results/2folds_50iterationsselectedFeatures.nparray.npy')

results5 = np.load('results/5folds_50iterationsresults.nparray.npy')
label5 = np.load('results/5folds_50iterationsselectedFeatures.nparray.npy')

results10 = np.load('results/10folds_50iterationsresults.nparray.npy')
labels10 = np.load('results/10folds_50iterationsselectedFeatures.nparray.npy')



scipy.io.savemat('results/2folds_50iterationsresults.mat', mdict={'arr': results2})
scipy.io.savemat('results/2folds_50iterationsselectedFeatures.mat', mdict={'arr': labels2})

scipy.io.savemat('results/5folds_50iterationsresults.mat', mdict={'arr': results5})
scipy.io.savemat('results/5folds_50iterationsselectedFeatures.mat', mdict={'arr': results5})

scipy.io.savemat('results/10folds_50iterationsresults.mat', mdict={'arr': results10})
scipy.io.savemat('results/10folds_50iterationsselectedFeatures.mat', mdict={'arr': labels10})


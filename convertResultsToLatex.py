import os
from fnmatch import fnmatch
from numpy import genfromtxt
import pandas as pd

#list all files
root = '/home/zeni/git/feature_selection/results'
pattern = "*.csv"

listOfFiles = []
names = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
        	listOfFiles.append(os.path.join(path, name))
        	names.append(name)

#print listOfFiles  	
#print names  	

file = open(root + "/magic.txt","a") 
#convert each file to latex
for f in range(0, len(listOfFiles)):
	actualFile = listOfFiles[f]
	#print file
	df = pd.read_csv(actualFile)
	data = df.as_matrix()
	data = data[:,1:]

	file.write('\\begin{table}[H]\n')
	file.write('\\centering\n')
	file.write('\\caption{')
	file.write(names[f].replace('.csv','').replace('_',' '))	
	file.write('}\n')
	file.write('\\label{')
	file.write(names[f])
	file.write('}\n')

	file.write('\\resizebox{\\columnwidth}{!}{\n')
	file.write('\\begin{tabular}{|')
	for i in range(0, data[0].shape[0] +1):
		file.write('l|')

	file.write('}\n')
	file.write('\\hline\n')


	file.write('\\begin{tabular}[c]{@{}l@{}}Features/\\\\ Classifiers\\end{tabular} &')
	
	file.write(' ' + data[0][1].replace('_','\\_') )

	for i in range(2, data[0].shape[0]):
		file.write('& \\begin{tabular}[c]{@{}l@{}}')
		labels = data[0][i].replace('_','\\_').split(', ')
		for j in range(0, len(labels)):
			file.write(labels[j])
			if j < len(labels)-1:
				file.write(', \\\\ ')
	
		file.write('\\end{tabular}')

	#print data[0].shape[0]
	file.write('\\\\ \\hline\n')
	

	for i in range(1, data.shape[0]):
		file.write( ' & '.join([''.join(row) for row in  data[i][:]]) + '\\\\ \\hline\n')


	file.write('\\end{tabular}\n')
	file.write('}\n')
	file.write('\\end{table}\n')
	file.write('\n\n\n')
file.close() 
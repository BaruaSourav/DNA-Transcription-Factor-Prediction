import pandas as pd 
import csv
import sys
import scipy.stats as stats

## Reading the file names from command-line arguments
## X_train.txt
## Y_train.txt
for args in sys.argv:
    print(str(args))

X_train_file_name = sys.argv[1]
Y_train_file_name = sys.argv[2]
X_train = pd.read_csv(X_train_file_name,sep='\t',header=None)
Y_train = pd.read_csv(Y_train_file_name,sep='\t')

## adding column name on X_train dataframe 
X_train.columns = ['TF_Name','TF_Seq']





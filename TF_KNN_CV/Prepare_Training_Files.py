import pandas as pd 
import csv
import sys


df = pd.read_csv("TF_sequences.txt",sep='\t', lineterminator='\n',header=None)
## Preparing the X_unseen.txt and X_train.txt data
X_train = df 
X_train.to_csv("X_train.txt", sep="\t", header=False,index=False)

## Preparing the Y_train.txt and Y_validation data
df = pd.read_csv("TF_output.txt",sep='\t', lineterminator='\n')
Y_train.to_csv("Y_train.txt", sep="\t", index=False,float_format='%.4f')

import pandas as pd 
import csv
import sys
import scipy.stats as stats

## Distance Function 1
def CalculateDistance(aminoAcidSeq1, aminoAcidSeq2):
    #print(aminoAcidSeq1,aminoAcidSeq2)
    distance = 0
    try:
        if aminoAcidSeq1[2] != aminoAcidSeq2[2]:
            distance+=1
        if aminoAcidSeq1[4] != aminoAcidSeq2[4]:
            distance+=1
        if aminoAcidSeq1[5] != aminoAcidSeq2[5]:
            distance+=1
        if aminoAcidSeq1[24] != aminoAcidSeq2[24]:
            distance+=1
        if aminoAcidSeq1[30] != aminoAcidSeq2[30]:
            distance+=1
        if aminoAcidSeq1[43] != aminoAcidSeq2[43]:
            distance+=1
        if aminoAcidSeq1[45] != aminoAcidSeq2[45]:
            distance+=1
        if aminoAcidSeq1[46] != aminoAcidSeq2[46]:
            distance+=1
        if aminoAcidSeq1[47] != aminoAcidSeq2[47]:
            distance+=1
        if aminoAcidSeq1[49] != aminoAcidSeq2[49]:
            distance+=1
        if aminoAcidSeq1[50] != aminoAcidSeq2[50]:
            distance+=1
        if aminoAcidSeq1[52] != aminoAcidSeq2[52]:
            distance+=1
        if aminoAcidSeq1[53] != aminoAcidSeq2[53]:
            distance+=1
        if aminoAcidSeq1[54] != aminoAcidSeq2[54]:
            distance+=1
        if (aminoAcidSeq1[56] != aminoAcidSeq2[56]):
            distance+=1
    except IndexError:
        print("Amino Acid Sequence length less than 57 characters")  
    return distance

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





import pandas as pd 
import csv
import sys

###########################
## Preparing the primary data set
###########################
# df = pd.read_csv("TF_sequences.txt",sep='\t', lineterminator='\n',header=None)
# ## Preparing the X_unseen.txt and X_train.txt data
# X_unseen = df.iloc[-30:] #splitting last 30 rows for the X_unseen.txt data 
# X_unseen.to_csv("X_unseen.txt", sep="\t", header=False,index=False)
# X_train = df.iloc[:-30] #splitting first 120 rows from the sequences
# X_train.to_csv("X_train.txt", sep="\t", header=False,index=False)

# ## Preparing the Y_train.txt and Y_validation data
# df = pd.read_csv("TF_output.txt",sep='\t', lineterminator='\n')
# df.rename(columns={"Vsx1\r": "Vsx1"},inplace=True) #ughhh faulty column name
# Y_validation = df.iloc[:, -30:] 
# Y_train = df.iloc[:,:-30]
# print(Y_train.head(17))
# Y_train.to_csv("Y_train.txt", sep="\t", index=False,float_format='%.4f')
# Y_validation.to_csv("Y_validation.txt", sep="\t", index=False, float_format='%.4f')
#####################################################################################


# Calculate distance of two given amino acid sequences
def CalculateDistance(aminoAcidSeq1, aminoAcidSeq2):
    distance = 0
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
    if aminoAcidSeq1[56] != aminoAcidSeq2[56]:
        distance+=1
    return distance
# print(CalculateDistance("MLRRAVFSDVQRKALEKTFQKQKYISKPDRKKLASKLGLKDSQVKIWFQNRRMKWRN","RKPRTIYSSYQLAALQRRFQKAQYLALPERAELAAQLGLTQTQVKIWFQNRRSKFKK"))
# print("Reading arguments")
for args in sys.argv:
    #print("test")
    print(str(args))
X_train_file_name = sys.argv[1]
Y_train_file_name = sys.argv[2]
X_unseen_file_name = sys.argv[3]
X_train = pd.read_csv(X_train_file_name,sep='\t', lineterminator='\n',header=None)
Y_train = pd.read_csv(Y_train_file_name,sep='\t', lineterminator='\n')
X_unseen = pd.read_csv(X_unseen_file_name,sep='\t',lineterminator='\n',header=None)

#print(X_unseen.iloc[0,1])
#print(X_train[1])
Y_predicted = pd.DataFrame()
for index,unseen_instance in X_unseen.iterrows():
    X_train['Distance'] = X_train.apply (lambda row: CalculateDistance(row[1],unseen_instance[1]), axis=1 )
    #CalculateDistance(X_train[1],X_unseen.iloc[0,1])
    X_train = X_train.sort_values(by='Distance')
    neighbours = X_train.head(15)[0]
    # print(neighbours.tolist())
    Y_train.rename(columns={"Pou1f1\r": "Pou1f1"},inplace=True)
    neighbour_vectors = Y_train[neighbours.tolist()]
    neighbour_vectors['Mean']= neighbour_vectors.mean(axis=1)
    #Y_predicted.index = neighbour_vectors.index
    #Y_predicted[row[0]] = neighbour_vectors['Mean']
    Y_predicted[unseen_instance[0]] = neighbour_vectors['Mean']
    print(neighbour_vectors)
    print(Y_predicted)
Y_predicted.to_csv("Y_predicted.txt", sep="\t", index=False,float_format='%.4f')



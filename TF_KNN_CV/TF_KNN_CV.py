# -*- coding: utf-8 -*-


import pandas as pd 
import csv
import sys
import numpy as np
import decimal 
import scipy.stats as stats

from sklearn.model_selection import KFold



pd.set_option('display.max_colwidth', -1)


## Distance Function 1
def CalculateDistance1(aminoAcidSeq1, aminoAcidSeq2):
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
        if (len(aminoAcidSeq1)>54 & len(aminoAcidSeq2)>54):
            if aminoAcidSeq1[54] != aminoAcidSeq2[54]:
                distance+=1
        if (len(aminoAcidSeq1)>56 & len(aminoAcidSeq2)>56):
            if (aminoAcidSeq1[56] != aminoAcidSeq2[56]):
                distance+=1
    except IndexError:
        print('Amino ACID 1: ',aminoAcidSeq1,' length: '+str(len(aminoAcidSeq1)))
        print('Amino ACID 2: ',aminoAcidSeq2,' length: '+str(len(aminoAcidSeq2)))
    return distance

def CalculateDistance2(aminoAcidSeq1,aminoAcidSeq2):
    intersection_cardinality = len(set.intersection(*[set(aminoAcidSeq1),set(aminoAcidSeq2)]))
    union_cardinality = len(set.union(*[set(aminoAcidSeq1),set(aminoAcidSeq2)]))
    return intersection_cardinality/float(union_cardinality)

## Grid Search Algorithm for performance dataframe
def GridSearchForModelSelection(performance_grid):
    print("Grid Search Method: ")
    print(performance_grid)
    selected_model= ''
    max_spearmans_corel_d1 = 0
    max_spearmans_corel_d2 = 0
    
    for index,row in performance_grid.iterrows():
        spearmans_corel_d1 = float(row['D1'].split()[0])
        spearmans_corel_d2 = float(row['D2'].split()[0])
        #print(spearmans_corel_d2)
        # print(spearmans_corel_d1)
        if  spearmans_corel_d1 > max_spearmans_corel_d1:
            max_spearmans_corel_d1 = spearmans_corel_d1
            d1_index = index
        if spearmans_corel_d2 > max_spearmans_corel_d2:
            max_spearmans_corel_d2 = spearmans_corel_d2
            d2_index = index
        #print(row['D1'].split())
    print(max_spearmans_corel_d1)
  
    print(max_spearmans_corel_d2)
    if max_spearmans_corel_d1 > max_spearmans_corel_d2:
        distance_function = 'D1'
        model_index = d1_index  
        print(max_spearmans_corel_d1)
    else:
        print(max_spearmans_corel_d2)
        model_index = d2_index  
        distance_function = 'D2'
    
    outputString = "Model Chosen: K=" + str(performance_grid.loc[model_index]['k'])+ ', '+distance_function
    file_object = open('model_selection_table.txt', 'a')
    file_object.write(outputString)
 
    # Close the file
    file_object.close()
    print(outputString)
    
    
#GridSearchForModelSelection(performance_grid)
#print (CalculateDistance2("RKARTAFSDHQLNQLERSFERQKYLSVQDRMDLAAALNLTDTQVKTWYQNRRTKWKR","RRNRFKWGPASQQILFQAYERQKSPSKEERETLVEECNRAECIQRGVSPSQAQGLGSNLVTEVRVYNWFANRRKEEAF"))

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



cv = KFold(n_splits=5)
performance_values = pd.DataFrame(columns=['standard_deviationD1','mean_spearmans_corelD1','standard_deviationD2','mean_spearmans_corelD2'])
Y_predicted = pd.DataFrame()

performance_grid = pd.DataFrame(columns=['k','D1','D2'])
k_values = [7,8,9]

for k in k_values:
    for train_index,test_index in cv.split(X_train):
        spearmans_corel_d1= 0
        standard_deviation_d1=0
        spearmans_corel_d2 = 0
        standard_deviation_d2 = 0
        for i in test_index:
            X_train_CV = X_train.loc[train_index,:]
            X_test_CV = X_train.loc[test_index,:]
            # model using distance function 1
            #X_train_CV
            X_train_CV['Distance1'] = X_train.apply(lambda row: CalculateDistance1(row['TF_Seq'],X_test_CV.loc[i]['TF_Seq']),axis=1)
            X_train_CV = X_train_CV.sort_values(by='Distance1')
            neighbours = X_train_CV.head(k)['TF_Name']
            # select the output vectors from the Y_train dataframe
            neighbour_vectors = Y_train[neighbours.tolist()]
            neighbour_vectors[X_test_CV.loc[i]['TF_Name'] + '(Predicted D1)']= neighbour_vectors.mean(axis=1)
            neighbour_vectors[X_test_CV.loc[i]['TF_Name']] = Y_train[X_test_CV.loc[i]['TF_Name']]
            #print(neighbour_vectors)
            #Y_predicted[neighbour_vectors[X_test_CV.loc[i]['TF_Name']]] = neighbour_vectors[X_test_CV.loc[i]['TF_Name'] + '(Predicted D1)']
            mean_std_predicted_values = neighbour_vectors.loc[:, neighbour_vectors.columns != X_test_CV.loc[i]['TF_Name']].std(axis=1)
            #standard_deviation_d1+= np.std(neighbour_vectors[X_test_CV.loc[i]['TF_Name']])
            standard_deviation_d1+= mean_std_predicted_values.mean() 
            #print (mean_std_predicted_values.mean())

            spearmans_corel_d1 += stats.spearmanr(neighbour_vectors[X_test_CV.loc[i]['TF_Name']],neighbour_vectors[X_test_CV.loc[i]['TF_Name'] + '(Predicted D1)'])[0]
            ############################################################################################33
            # Removing distance1 column and adding distance2 column and sorting
            X_train_CV.drop(columns=['Distance1'],inplace=True)
            X_train_CV['Distance2'] = X_train.apply(lambda row: CalculateDistance2(row['TF_Seq'],X_test_CV.loc[i]['TF_Seq']),axis=1)
            X_train_CV = X_train_CV.sort_values(by='Distance2')
            neighbours = X_train_CV.head(k)['TF_Name']
            neighbour_vectors = Y_train[neighbours.tolist()]
            neighbour_vectors[X_test_CV.loc[i]['TF_Name'] + '(Predicted D2)']= neighbour_vectors.mean(axis=1)
            neighbour_vectors[X_test_CV.loc[i]['TF_Name']] = Y_train[X_test_CV.loc[i]['TF_Name']]
            mean_std_predicted_values = neighbour_vectors.loc[:, neighbour_vectors.columns != X_test_CV.loc[i]['TF_Name']].std(axis=1)

            standard_deviation_d2+= mean_std_predicted_values.mean() 
            spearmans_corel_d2 += stats.spearmanr(neighbour_vectors[X_test_CV.loc[i]['TF_Name']],neighbour_vectors[X_test_CV.loc[i]['TF_Name'] + '(Predicted D2)'])[0]

            #print(neighbour_vectors)

        mean_standard_deviation_d1 = standard_deviation_d1/test_index.size
        mean_standard_deviation_d2 = standard_deviation_d2/test_index.size
        #mean_standard_deviation_d1    #print (Y_predicted)

        mean_spearman_cor_d1 = spearmans_corel_d1/test_index.size
        mean_spearman_cor_d2 = spearmans_corel_d2/test_index.size

        performance_row =  pd.DataFrame([[mean_standard_deviation_d1, mean_spearman_cor_d1,mean_standard_deviation_d2,mean_spearman_cor_d2]],columns=['standard_deviationD1','mean_spearmans_corelD1','standard_deviationD2','mean_spearmans_corelD2'])
        performance_values = performance_values.append(performance_row, ignore_index = True)

        print('Mean Spearman COR :',mean_spearman_cor_d1) 

    #print('Spearman COR:')
    print('Performance Dataframe:\n', performance_values)
    meanSpearmand1 = round(performance_values['mean_spearmans_corelD1'].mean(),4)
    meanSpearmand2 = round(performance_values['mean_spearmans_corelD2'].mean(),4)

    meanSD1 = round(performance_values['standard_deviationD1'].mean(),4)
    meanSD2 = round(performance_values['standard_deviationD2'].mean(),4)

    #print(test)

    performance_grid_row =  pd.DataFrame([[k,str(meanSpearmand1) +' ± '+ str(meanSD1),str(meanSpearmand2)+' ± '+str(meanSD2)]],columns=['k','D1','D2'])
    performance_grid = performance_grid.append(performance_grid_row,ignore_index = True)

    print('Performance Grid \n')
    print(performance_grid)
performance_grid.to_csv("model_selection_table.txt", sep="\t",index=False)

GridSearchForModelSelection(performance_grid)




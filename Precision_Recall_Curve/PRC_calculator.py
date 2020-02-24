import pandas as pd 
import csv
import sys
import numpy as np
import decimal 
import scipy.stats as stats

def get_TP_count(class_prediction_dataframe):
    return len(class_prediction_dataframe[(class_prediction_dataframe['Predicted Class'] == 1) & (class_prediction_dataframe['Actual Class'] == 1)])
def get_FP_count(class_prediction_dataframe):
    return len(class_prediction_dataframe[(class_prediction_dataframe['Predicted Class'] == 1) & (class_prediction_dataframe['Actual Class'] == 0)])
def get_FN_count(class_prediction_dataframe):
    return len(class_prediction_dataframe[(class_prediction_dataframe['Predicted Class'] == 0) & (class_prediction_dataframe['Actual Class'] == 1)])


def main():
    prediction_result_file_name = sys.argv[1]  
    class_predictions = pd.read_csv(prediction_result_file_name,sep='\t',header=None) 
    class_predictions.columns = ['Confidence Values','Actual Class']
    pr_pairs = pd.DataFrame(columns=['Threshold','Precision','Recall'])
    threshold_values = np.arange(0.4,0.9,0.025)
    
    for threshold in threshold_values:
        print(threshold)
        if 'Predicted Class' in class_predictions.columns:
            class_predictions.drop('Predicted Class',axis=1, inplace = True)
        class_predictions['Predicted Class'] = class_predictions.apply(lambda row: 1 if row['Confidence Values'] >=threshold else 0,axis=1)
        #print(class_predictions)
        TP = get_TP_count(class_predictions)
        FP = get_FP_count(class_predictions)
        FN = get_FN_count(class_predictions)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        pr_pair_row = pd.DataFrame([[threshold,precision,recall]], columns=['Threshold','Precision','Recall'])
        pr_pairs = pr_pairs.append(pr_pair_row,ignore_index= True)
        print('Precision : ',precision)
        print('Recall : ', recall)



if __name__ == '__main__':
    main()
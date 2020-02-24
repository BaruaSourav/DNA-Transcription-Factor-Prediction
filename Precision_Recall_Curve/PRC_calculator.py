import pandas as pd 
import csv
import sys
import numpy as np
import decimal 
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

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
    print("Writing PR_table.txt file....")    
    pr_pairs.to_csv("PR_table.txt", sep="\t",index=False,float_format="%.4f")
    
    ### Constructing PR Curve
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = pr_pairs['Recall'],
            y = pr_pairs['Precision'],
            mode ="lines+markers",
            line_shape='vh',
            text='asdasd'
        )
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title="Precision - Recall Curve",
        margin=dict(l=20, r=150, t=150, b=20),
        xaxis_title="Recall",
        yaxis_title="Precision",
        font=dict(
            size=18,
            color="#7f7f7f"
        ),
        width=800,
        height=700,
        
    )
    print("Writing the PRC.png file.....")
    #fig.show()
    fig.write_image("PRC.png")
    print("Successfully created the PRC.png plot file and the threshold-precision-recall table on PR_table.txt file")



if __name__ == '__main__':
    main()
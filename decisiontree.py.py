import csv
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt  

import pickle


#start = time.time()
# Timing Start

# Function to check, if node is unique to make classification
def unique_data_labels(data):
    count=len(np.unique(data[:,-1]))
    if(count==1):
        return 1
    else:
        return 0
# Function to Make classifciation, based on majority classification model
def classify_data_labels(data):
    array,counts=np.unique(data[:,-1],return_counts=True)
    tmp=np.argmax(counts)
    predicted_label=array[tmp]
    
    return predicted_label
# Function to calucate overall system entropy
def data_entropy(data):
    array,counts=np.unique(data[:,-1],return_counts=True)
    Pr_array=counts/float(np.sum(counts))
    entropy=-np.sum(Pr_array*(np.log2(Pr_array)))

    return entropy
# Function to compute expected entropy
def expected_entropy(data_left,data_right):
    p_left=float(len(data_left))/(len(data_left)+len(data_right))
    p_right=float(len(data_right))/(len(data_left)+len(data_right))
    exp_entropy=(p_left*data_entropy(data_left) + p_right*data_entropy(data_right))
    
    return exp_entropy
# Function for information gain
def information_gain(data,data_left,data_right):
    e1=data_entropy(data)
    e2=expected_entropy(data_left,data_right)
    IG=e1-e2
    
    return IG
# Function for splitting data to create binary tree
def create_data_split(data,split_col,split_val):
    split_col_values=data[:,split_col]
    data_left=data[split_col_values<=split_val]
    data_right=data[split_col_values>split_val]
    
    return data_left,data_right
# Function to estimate best split decision, based on Information gain ID3 algorithm
def find_best_decision(data):   
    split_choices={}
    
    for i in range(0,np.shape(data)[1]-1):
        split_choices[i]=[]
        unique_values=np.unique(data[:,i])
        for j in range(0,len(unique_values)):
            if(j!=0):
                val1=unique_values[j]
                val2=unique_values[j-1]
                split_val=(val1+val2)/2
                split_choices[i].append(split_val)

    IG=0
        
    for i in range(0,len(split_choices)):
        for j in split_choices[i]:
            data_left,data_right=create_data_split(data,i,j)
            IG_new=information_gain(data,data_left,data_right)      
            if(IG_new>=IG):
                IG=IG_new
                best_split_col=i
                best_split_val=j
   
    return best_split_col,best_split_val
# Confusion Matrix to calculate F1 score and accuracy results 
def CM(Y_pred,Y_true):
    Con_Mat=np.zeros((11,11))
    TP=np.zeros(11)
    FP=np.zeros(11)
    FN=np.zeros(11)
    F=np.zeros(11)
    
    for i in range(0,len(Y_pred)):
        Con_Mat[int(Y_true[i])][int(Y_pred[i])]=Con_Mat[int(Y_true[i])][int(Y_pred[i])]+1
        
    for i in range(0,11):
        for j in range(0,11):
            if(i==j):
                TP[i]=Con_Mat[i][j]
            else:
                FN[i]=FN[i]+Con_Mat[i][j]
                FP[i]=FP[i]+Con_Mat[j][i]
        if(TP[i]==0):
            F[i]=0
        else:
            F[i]=2*TP[i]/float(2*TP[i]+FP[i]+FN[i])
    
    F1_Score=float(np.sum(F))/(len(np.unique(Y_true))) 
    Accuracy=float(np.sum(TP))/(len(Y_pred))
    
    return Accuracy,F1_Score
# Function to generate tree , used in training and testing 
# The hyper parameters, Max depth and min instance count
def create_decision_treeID3(data,min_instance_count=2,Max_Depth=11,depth_count=0):
    
    if((unique_data_labels(data))or(len(data)<min_instance_count)or(depth_count==Max_Depth)):
        predicted_label=classify_data_labels(data)
        return predicted_label
    
    else:
        depth_count=depth_count+1       
        bsplit_col,bsplit_val=find_best_decision(data)
        data_left,data_right=create_data_split(data,bsplit_col,bsplit_val)
        decision="{} <= {}".format(bsplit_col,bsplit_val)
        sub_tree={decision:[]}
        
        left=create_decision_treeID3(data_left,min_instance_count,Max_Depth,depth_count)
        right=create_decision_treeID3(data_right,min_instance_count,Max_Depth,depth_count)
    
        if(left==right):
            sub_tree=left
        else:
            sub_tree[decision].append(left)
            sub_tree[decision].append(right)
            
        return sub_tree
# Function for Making decision to classify example for an instance    
def make_decision(x,tree):
    decision = list(tree.keys())[0]
    col_index, comparison_operator, value = decision.split(" ")

    if(x[int(col_index)]<=float(value)):
        label=tree[decision][0]
    else:
        label=tree[decision][1]
        
    if not isinstance(label, dict):
        return label
    else:
        return make_decision(x,label)
# Function to print final output after cross validation, similar to KNN, using Decision trees
def Final_Output_Test(X_train,X_test,K):
    X_val=X_train[0:np.shape(X_train)[0]/5]
    X_train=X_train[np.shape(X_train)[0]/5:np.shape(X_train)[0]]
    
    Y_train=X_train[:,11]
    Y_val=X_val[:,11]
    Y_test=X_test[:,11]
       
    Mean=X_train[:,0:11].mean(0)
    Std=X_train[:,0:11].std(0)    
    X_train[:,0:11]=(X_train[:,0:11]-Mean)/Std
    X_val[:,0:11]=(X_val[:,0:11]-Mean)/Std
    X_test[:,0:11]=(X_test[:,0:11]-Mean)/Std
    
    for k in range(K,K+1):
        tree = create_decision_treeID3(X_train,min_instance_count=2,Max_Depth=k,depth_count=0)
        
        Y_pred=np.zeros(len(X_train))
        for i in range(0,len(X_train)):
            Y_pred[i]=make_decision(X_train[i],tree) 
               
        TrAccuracy,TrF1_Score=CM(Y_pred,Y_train)
        
        Y_pred=np.zeros(len(X_val))
        for i in range(0,len(X_val)):
            Y_pred[i]=make_decision(X_val[i],tree) 
               
        VAccuracy,VF1_Score=CM(Y_pred,Y_val)

        Y_pred=np.zeros(len(X_test))
        for i in range(0,len(X_test)):
            Y_pred[i]=make_decision(X_test[i],tree) 
               
        TsAccuracy,TsF1_Score=CM(Y_pred,Y_test)

    return TsAccuracy,TsF1_Score,VAccuracy,VF1_Score,TrAccuracy,TrF1_Score    
    
# Function for Cross Validation, only parameter is Kmax- Max depth of Tree
def cross_validation_test(X_train,K_Max):
    X_val=X_train[0:np.shape(X_train)[0]/5]
    X_train=X_train[np.shape(X_train)[0]/5:np.shape(X_train)[0]]
    
    Y_train=X_train[:,11]# Splitting data
    Y_val=X_val[:,11]
    
    Mean=X_train[:,0:11].mean(0)# Normalizing data
    Std=X_train[:,0:11].std(0)
    X_train[:,0:11]=(X_train[:,0:11]-Mean)/Std
    X_val[:,0:11]=(X_val[:,0:11]-Mean)/Std    
    
    Accuracy=np.zeros(K_Max+1)
    F1_Score=np.zeros(K_Max+1)
   
    for k in range(2,K_Max+1): # Cross Validation
        Y_pred=np.zeros(len(X_val)) # Generating tree command below with varying depth
        tree = create_decision_treeID3(X_train,min_instance_count=2,Max_Depth=k,depth_count=0)
        for i in range(0,len(X_val)):
            Y_pred[i]=make_decision(X_val[i],tree)
        
        Accuracy[k],F1_Score[k]=CM(Y_pred,Y_val) # Computin F1 score and accuracy
    
        print("The value of Max-Depth is %d ." %(k))
        print(F1_Score[k])

    return Accuracy,F1_Score


file = open('winequality-white.csv')

data=[]
TsAc=[]
TsF1=[]
VAc=[]
VF1=[]
TrAc=[]
TrF1=[]

for row in file:
    a=row.split(';')
    data.append(a)

del data[0]

X=np.asarray(data).astype('float')
#np.random.seed(5)
np.random.shuffle(X)

X_1=X[0:np.shape(X)[0]/4]
X_2=X[np.shape(X)[0]/4:2*(np.shape(X)[0]/4)]
X_3=X[2*(np.shape(X)[0]/4):3*(np.shape(X)[0]/4)]
X_4=X[3*(np.shape(X)[0]/4):np.shape(X)[0]]

test=[X_1,X_2,X_3,X_4]
tr1=np.concatenate((X_2,X_3,X_4),axis=0)
tr2=np.concatenate((X_3,X_4,X_1),axis=0)
tr3=np.concatenate((X_4,X_1,X_2),axis=0)
tr4=np.concatenate((X_1,X_2,X_3),axis=0)
train=[tr1,tr2,tr3,tr4]   # Data split into folds for cross validation and testing

K_best_final=14  # Best Max-Depth

print("Hyper-parameters:")
print("Best Max-Depth in Fold: %d" %(K_best_final))

for i in range(0,4):
    X_test=test[i]
    X_train=train[i]
    
    TsAccuracy,TsF1_Score,VAccuracy,VF1_Score,TrAccuracy,TrF1_Score=Final_Output_Test(X_train,X_test,K_best_final)
    TsAc.append(TsAccuracy)
    TsF1.append(TsF1_Score)
    VAc.append(VAccuracy)
    VF1.append(VF1_Score)
    TrAc.append(TrAccuracy)
    TrF1.append(TrF1_Score)

    print("Fold-%d:" %(i+1))
    print("Training: F1 Score: %f , Accuracy: %f" %(TrF1_Score,TrAccuracy))
    print("Validation: F1 Score: %f , Accuracy: %f" %(VF1_Score,VAccuracy))
    print("Test: F1 Score: %f , Accuracy: %f" %(TsF1_Score,TsAccuracy))
    

print("Average:")
print("Training: F1 Score: %f , Accuracy: %f" %(np.mean(TrF1),np.mean(TrAc)))
print("Validation: F1 Score: %f , Accuracy: %f" %(np.mean(VF1),np.mean(VAc)))
print("Test: F1 Score: %f , Accuracy: %f" %(np.mean(TsF1),np.mean(TsAc)))

# Timing Metrics
#end = time.time()
#print("The time taken for the algorithm computation is :- %f seconds." % (end-start))

#file = open('winequality-white.csv')
#
#data=[]
#Ac=[]
#F1=[]
#
#TsAc=[]
#TsF1=[]
#VAc=[]
#VF1=[]
#TrAc=[]
#TrF1=[]
#
#for row in file:
#    a=row.split(';')
#    data.append(a)
#
#del data[0]
#
#X=np.asarray(data).astype('float')
##np.random.seed(0)
#np.random.shuffle(X)
#
#X_1=X[0:np.shape(X)[0]/4]
#X_2=X[np.shape(X)[0]/4:2*(np.shape(X)[0]/4)]
#X_3=X[2*(np.shape(X)[0]/4):3*(np.shape(X)[0]/4)]
#X_4=X[3*(np.shape(X)[0]/4):np.shape(X)[0]]
#
#test=[X_1,X_2,X_3,X_4]
#tr1=np.concatenate((X_2,X_3,X_4),axis=0)
#tr2=np.concatenate((X_3,X_4,X_1),axis=0)
#tr3=np.concatenate((X_4,X_1,X_2),axis=0)
#tr4=np.concatenate((X_1,X_2,X_3),axis=0)
#train=[tr1,tr2,tr3,tr4]
#
#for i in range(0,4):
#    X_test=test[i]
#    X_train=train[i]
#    
#    Accuracy,F1_Score=cross_validation_test(X_train,25)
#    K_best_fold=np.argmax(F1_Score)
#    print("The best value of Max-Depth is %d and fold number is %d." % (K_best_fold,i+1))
#    print(Accuracy[K_best_fold])
#    print(F1_Score[K_best_fold])
#
#    x = np.arange(2,26, 1)
#    Accuracy=Accuracy[2:]
#    F1_Score=F1_Score[2:]
#    F1.append(F1_Score)
#    Ac.append(Accuracy)
#    
#    plt.figure(1)
#    plt.plot(x,Accuracy, label = "fold %d" %(i+1))
#    plt.figure(2)
#    plt.plot(x,F1_Score, label = "fold %d" %(i+1))
#    
#    TsAccuracy,TsF1_Score,VAccuracy,VF1_Score,TrAccuracy,TrF1_Score=Final_Output_Test(X_train,X_test,K_best_fold)
#    TsAc.append(TsAccuracy)
#    TsF1.append(TsF1_Score)
#    VAc.append(VAccuracy)
#    VF1.append(VF1_Score)
#    TrAc.append(TrAccuracy)
#    TrF1.append(TrF1_Score)
#
#    print("Hyper-parameters:")
#    print("Best Max-Depth in Fold: %d" %(K_best_fold))
#
#    print("Fold-%d:" %(i+1))
#    print("Training: F1 Score: %f , Accuracy: %f" %(TrF1_Score,TrAccuracy))
#    print("Validation: F1 Score: %f , Accuracy: %f" %(VF1_Score,VAccuracy))
#    print("Test: F1 Score: %f , Accuracy: %f" %(TsF1_Score,TsAccuracy))
#    
# 
#print("Average:")
#print("Training: F1 Score: %f , Accuracy: %f" %(np.mean(TrF1),np.mean(TrAc)))
#print("Validation: F1 Score: %f , Accuracy: %f" %(np.mean(VF1),np.mean(VAc)))
#print("Test: F1 Score: %f , Accuracy: %f" %(np.mean(TsF1),np.mean(TsAc)))
#    
#
#plt.figure(1)
#plt.xlabel('K') 
## naming the y axis 
#plt.ylabel('Accuracy') 
## giving a title to my graph 
#plt.title('Accuracy vs K')   
## show a legend on the plot 
#plt.legend()   
## function to show the plot 
#plt.savefig('Accuracy.png')
#
#plt.figure(2)
#plt.xlabel('K') 
## naming the y axis 
#plt.ylabel('F1 Score') 
## giving a title to my graph 
#plt.title('F1 Scores vs K')   
## show a legend on the plot 
#plt.legend()   
## function to show the plot 
#plt.savefig('F1_Score.png')
#
#pickle.dump(X, open( "X_data_saved.p", "wb" ) ) 
#pickle.dump(Ac, open( "Ac_data_saved.p", "wb" ) )
#pickle.dump(F1, open( "F1_data_saved.p", "wb" ) )
#
#
#end = time.time()
#print("The time taken for the algorithm computation is :- %f seconds." % (end-start))

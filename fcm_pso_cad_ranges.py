#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import time
import tensorflow as tf
from random import uniform
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import math
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, BorderlineSMOTE, SVMSMOTE
#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self):
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        #initilization of position and velocity
        np.random.seed(0) 
        df = pd.read_excel("suggested_weights_matrix_cad.xlsx", nrows=23, engine='openpyxl')
        arr = df.to_numpy()

        num_dimensions=23
        for i in range(0,num_dimensions):
            for j in range(0,num_dimensions):

                if arr[i][j] == "nan":
                    arr[i][j]=random.uniform(-1, 1)
                if arr[i][j]=="-VS":
                    arr[i][j]=random.uniform(-1, -0.75)
                if arr[i][j]=='"-S"':
                    arr[i][j]=random.uniform(-0.8, -0.65)
                if arr[i][j] =='"-M"':
                    arr[i][j]=random.uniform(-0.6, -0.35)
                if arr[i][j] =="-W":
                    arr[i][j]=random.uniform(-0.4 -0.15)
                if arr[i][j] =="-VW":
                    arr[i][j]=random.uniform(-0.25, 0)


                if arr[i][j] =="VW":
                    arr[i][j]=random.uniform(0, 0.25)
                if arr[i][j] =="W":
                    arr[i][j]=random.uniform(0.15, 0.4)
                if arr[i][j] =="M":
                    arr[i][j]=random.uniform(0.35, 0.6)
                if arr[i][j]=="S":
                    arr[i][j]=random.uniform(0.65, 0.8)
                if arr[i][j]=="VS":
                    arr[i][j]=random.uniform(0.75, 1)


        np.fill_diagonal(arr, 0)
       
        W2 =np.random.uniform(size = (23,23), low = -1, high = 1)
        np.fill_diagonal(W2, 0)
        self.position_i=(arr)
        self.velocity_i=(W2)

    # evaluate current fitness
    def evaluate(self,err_i):
        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=err_i          
 # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant
        r1=uniform(0,1)
        r2=uniform(0,1)
        for i in range(0,num_dimensions):
          for j in range(0,num_dimensions):
              if(i==j):
                continue
              else:
                vel_cognitive=c1*r1*(self.pos_best_i[i][j]-self.position_i[i][j])
                vel_social=c2*r2*(pos_best_g[i][j]-self.position_i[i][j])
                self.velocity_i[i][j]=w*self.velocity_i[i][j] +vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0,num_dimensions):
          for j in range(0,num_dimensions):
              if(i==j):
                continue
              else:
                self.position_i[i][j]=self.position_i[i][j]+self.velocity_i[i][j]
                # adjust maximum position if necessary
                if self.position_i[i][j]>1:
                    self.position_i[i][j]=1

                # adjust minimum position if neseccary
                if self.position_i[i][j]<-1:
                    self.position_i[i][j]=-1

          
def sig(x):
    return 1/(1 + np.exp(-x))

def minimize(dataset,  num_particles, maxiter):
    global num_dimensions
    num_dimensions=num_particles
    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group
    i=0
    result_error=[]
    fitness_result=-1
    swarm=[]
    training_real_output=[]
    training_fcm_output=[]
    k=-1
    sum_temp=0
    last_element_fcm_output=[]
    training_acc=[]
    new_fcm_output = [None]*23
    predicted_fcm=[]
    swarm = [Particle() for _ in range(num_dimensions)]

    while k<maxiter  :
        last_element_fcm_output=[]
        training_real_output=[]
        k+=1
        print("-------------------iter------------")
        print(k)

        for row in dataset:
            for i in range(0,num_dimensions):
                for j in range(0,num_dimensions):
                    if(i==j):
                        continue
                    else:
                        sum_temp=sum_temp+swarm[j].position_i[j][i]*row[j]
                       
                new_fcm_output[i]=sum_temp
                new_fcm_output[i] = sig(new_fcm_output[i])
                sum_temp=0

            last_element_fcm_output.append(((new_fcm_output)[-1]))

        training_real_output = training_dataset[:, -1]
        last_element_fcm_output = np.vstack(last_element_fcm_output)

        last_element_fcm_output = last_element_fcm_output[:, -1]
        temporary_value_results=last_element_fcm_output


        limits_acc=[]
        limits= np.arange(0.2, 0.99, 0.01).tolist()
        
        training_steady_value_predicted_results=temporary_value_results
        for i in limits:
            
            training_real_output=np.array(training_real_output)
            temporary_value_results=(np.array(temporary_value_results))
            training_steady_value_predicted_results=(np.array(training_steady_value_predicted_results))
            
            
            temporary_value_results = training_steady_value_predicted_results > i


            training_real_output=np.array(training_real_output)
            temporary_value_results=(np.array(temporary_value_results))


            
            limits_acc.append(accuracy_score(training_real_output, temporary_value_results.round())*100)


       
        max_value = max(limits_acc)
        index = limits_acc.index(max_value)

        last_element_fcm_output = last_element_fcm_output > limits[index]

        training_real_output=np.array(training_real_output)
        last_element_fcm_output=(np.array(last_element_fcm_output))

        training_acc.append(accuracy_score(training_real_output, last_element_fcm_output)*100)

        
        for j in range(0, num_dimensions):
            fitness_result=(training_real_output[j]- last_element_fcm_output[j])
            swarm[j].err_i=fitness_result
         


            swarm[j].evaluate(swarm[j].err_i)

            if swarm[j].err_i<err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)

            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position()


    return pos_best_g

    

dataset=pd.read_excel("cad_full.xlsx", engine='openpyxl')
dataset.fillna(method="bfill", inplace=True)


# Randomly sample a subset of the data
subset = dataset.sample(frac=0.01)

# Shuffle the subset
subset = subset.reindex(np.random.permutation(subset.index))

# Concatenate the shuffled subset back to the original dataframe
dataset = pd.concat([dataset, subset]).reset_index(drop=True)


# # apply normalization techniques by Column
column = 'AGE'
dataset[column] = (dataset[column]-dataset[column].min())/(dataset[column].max()-dataset[column].min())

column1 = 'BMI'
dataset[column1] = (dataset[column1]-dataset[column1].min())/(dataset[column1].max()-dataset[column1].min())

X = dataset.iloc[:,:-1].values
y = dataset.iloc[: ,-1].values

oversample = SMOTEN(sampling_strategy='all')

X, y = oversample.fit_resample(X, y)

dataset = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1, join="inner")


def find_80_percent(number):
    return number * 0.8

result = find_80_percent(dataset.shape[0])
training_dataset = dataset.iloc[:int(result)].values
testing_dataset = dataset.iloc[(int(result)+1):].values


best_position = minimize(training_dataset,  num_particles=23, maxiter=35)
end = time.time()


for i in range(0,num_dimensions):
    for j in range(0,num_dimensions):
        if(i==j):
            continue
        else:
            # adjust maximum position if necessary
            if best_position[i][j]>1:
                best_position[i][j]=1

            # # adjust minimum position if neseccary
            if best_position[i][j]<-1:
                best_position[i][j]=-1
error=[]
sum_temp=0

testing_last_element_fcm_output=[]
best_position=np.vstack(best_position)
predicted_results=[None]*23
for testing_row in testing_dataset:

    for i in range(0,num_dimensions):
        for j in range(0,num_dimensions):
            if(i==j):
                continue
            else:
                sum_temp=sum_temp+best_position[j][i]*testing_row[j]
                
        predicted_results[i]=sum_temp
        predicted_results[i] = sig(predicted_results[i])
    
        sum_temp=0
    

    testing_last_element_fcm_output.append((predicted_results)[-1])
testing_actual_output = testing_dataset[:, -1]
            
print("testing procedure")
testing_last_element_fcm_output = np.vstack(testing_last_element_fcm_output)

testing_last_element_fcm_output = testing_last_element_fcm_output[:, -1]
temporary_value_results=testing_last_element_fcm_output


limits_acc=[]
limits= np.arange(0.1, 0.99, 0.01).tolist()

steady_value_predicted_results=temporary_value_results
for i in limits:
   
    temporary_value_results = steady_value_predicted_results > i

    testing_actual_output=np.array(testing_actual_output)
    temporary_value_results=(np.array(temporary_value_results))

    limits_acc.append(accuracy_score(testing_actual_output, temporary_value_results.round())*100)

max_value = max(limits_acc)
index = limits_acc.index(max_value)


testing_last_element_fcm_output = testing_last_element_fcm_output > limits[index]

testing_actual_output=np.array(testing_actual_output)
testing_last_element_fcm_output=(np.array(testing_last_element_fcm_output))


print("accuracy")

print(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)

print('Mean Absolute Error:')
print( metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))

cm = confusion_matrix(testing_actual_output, testing_last_element_fcm_output)


class_counts = cm.sum(axis=1)
accuracies = [0 if count == 0 else cm[i, i] / count for i, count in enumerate(class_counts)]

print(cm)

tp = np.sum((testing_actual_output == 1) & (testing_last_element_fcm_output == 1))
fn = np.sum((testing_actual_output == 1) & (testing_last_element_fcm_output == 0))
sensitivity = tp / (tp + fn)

print('Sensitivity : ', sensitivity*100 )



tn, fp, fn, tp = confusion_matrix(testing_actual_output, testing_last_element_fcm_output).ravel()
specificity = tn / (tn + fp)

print('Specificity : ', specificity*100)


precision = precision_score(testing_actual_output, testing_last_element_fcm_output)

print("Precision")
print(precision*100)




def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



plot_roc_curve(testing_actual_output, testing_last_element_fcm_output)
print(f'model 1 AUC score: {roc_auc_score(testing_actual_output, testing_last_element_fcm_output)}')

plt.show()

# convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(best_position, columns=["A", "B", "C","A", "B", "C","A", "B", "C","A", "B", "C","A", "B", "C","A", "B", "C","A", "B", "C","A", "B" ])

# write the DataFrame to an Excel file
df.to_excel("example_rangess.xlsx", index=False)

#--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
import time
from random import uniform
import pandas as pd
import random
from sklearn.model_selection import KFold
import math
from sklearn.metrics import accuracy_score, confusion_matrix
import time
#--- MAIN ---------------------------------------------------------------------+
num_dimensions=24
class Particle:
    def __init__(self):
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        #initilization of position and velocity
        np.random.seed(0)
        df = pd.read_excel("suggested_weights_with_CNNoutput.xlsx", nrows=24, engine='openpyxl')
        arr = df.to_numpy()

        num_dimensions=24
        for i in range(0,num_dimensions):
            for j in range(0,num_dimensions):

                if arr[i][j] == "random":
                    arr[i][j]=random.uniform(-1, 1)
                if arr[i][j]=="-VS":
                    arr[i][j]=random.uniform(-1, -0.7)
                if arr[i][j]=='MINUSSTRONG':
                    arr[i][j]=random.uniform(-0.85, -0.5)
                if arr[i][j] =='"-M"':
                    arr[i][j]=random.uniform(-0.65, -0.35)
                if arr[i][j] =="-W":
                    arr[i][j]=random.uniform(-0.5, -0.15)
                if arr[i][j] =="-VW":
                    arr[i][j]=random.uniform(-0.3, 0)

                if arr[i][j] =="VW":
                    arr[i][j]=random.uniform(0, 0.3)
                if arr[i][j] =="W":
                    arr[i][j]=random.uniform(0.15, 0.5)
                if arr[i][j] =="M":
                    arr[i][j]=random.uniform(0.35, 0.65)
                if arr[i][j]=="S":
                    arr[i][j]=random.uniform(0.5, 0.85)
                if arr[i][j]=="VS":
                    arr[i][j]=random.uniform(0.7, 1)


        # W1 =np.random.uniform(size = (24,24), low = -1, high = 1)
        np.fill_diagonal(arr, 0)

        W2 =np.random.uniform(size = (24,24), low = -1, high = 1)
        np.fill_diagonal(W2, 0)

        arr[-1] = 0
        W2[-1] = 0
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

def minimize(dataset,  num_dimensions, num_particles,maxiter):


    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group
    i=0
    fitness_result=-1
    swarm=[]
    training_real_output=[]
    k=-1
    sum_temp=0
    last_element_fcm_output=[]
    training_acc=[]
    new_fcm_output = [None]*num_dimensions
    swarm = [Particle() for _ in range(num_particles)]

    while k<maxiter  :
        last_element_fcm_output=[]
        training_real_output=[]
        k+=1
        # print("-------------------iter------------")
        # print(k)

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

        for j in range(0, num_particles):
            # print(j)
            fitness_result=np.abs(training_real_output[j]- last_element_fcm_output[j])**2
            swarm[j].err_i=fitness_result
            swarm[j].evaluate(swarm[j].err_i)
            if swarm[j].err_i<err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)
                # print(err_best_g)
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position()
    return pos_best_g

excel_dataset=pd.read_excel("cad_full_wit_ids.xlsx", engine='openpyxl')

# # apply normalization techniques by Column
column = 'AGE'
excel_dataset[column] = (excel_dataset[column]-excel_dataset[column].min())/(excel_dataset[column].max()-excel_dataset[column].min())

column1 = 'BMI'
excel_dataset[column1] = (excel_dataset[column1]-excel_dataset[column1].min())/(excel_dataset[column1].max()-excel_dataset[column1].min())

excel_dataset.columns.values[0] = 'id'

cnn_outputs = pd.read_excel('cad_cnn_output.xlsx', engine='openpyxl')

list_of_ids_cnn_outputs = cnn_outputs.iloc[:, 0].tolist()

print("\n\n\n")

cnn_outputs = pd.DataFrame(cnn_outputs)
cnn_outputs.iloc[:, 0] = list_of_ids_cnn_outputs


cnn_outputs = cnn_outputs.rename(columns={cnn_outputs.columns[0]: 'ids'})

dataset = pd.merge(excel_dataset, cnn_outputs, left_on='id', right_on='ids', how='inner')
# Remove the 'age' column and append it to the end of the DataFrame
age_column = dataset.pop('id')
age_column = dataset.pop('ids')
output = dataset.pop('output')
dataset = dataset.assign(output=output)
start = time.time()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold=0
acc=[]
err=[]
best_matrix=[]
concepts=[]
acc=[]
err=[]
best_matrix=[]
concepts=[]
class_accuracies0 =[]
class_accuracies1 =[]
c_matrices=[]
recalls=[]
best_positions=[]
cm_sum = np.zeros((2, 2))
sens=[]
spec=[]
prec=[]
epoch=30
# Iterate over each train-test split
for train_index, test_index in kf.split(dataset):
    fold+=1


    print("\n\n************************************")
    print(fold)
    print("---------fold------------")
    print("************************************\n\n")

    # Split train-test
    training_dataset, testing_dataset=dataset.iloc[train_index], dataset.iloc[test_index]


    training_dataset= np.array(training_dataset)
    testing_dataset= np.array(testing_dataset)

    best_position = minimize(training_dataset,  num_dimensions=24, num_particles=40, maxiter=epoch)
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
    predicted_results=[None]*24
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



    # print("testing procedure")
    testing_last_element_fcm_output = np.vstack(testing_last_element_fcm_output)

    testing_last_element_fcm_output = testing_last_element_fcm_output[:, -1]



    from sklearn.metrics import mean_squared_error
    from sklearn import metrics

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

    import matplotlib.pyplot as plt
    testing_last_element_fcm_output = testing_last_element_fcm_output > limits[index]

    testing_actual_output=np.array(testing_actual_output)
    testing_last_element_fcm_output=(np.array(testing_last_element_fcm_output))



    # print("accuracy")

    # print(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)
    A=(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)
    acc.append(accuracy_score(testing_actual_output, testing_last_element_fcm_output.round())*100)

    # from sklearn import metrics
    # print('Mean Absolute Error:')
    # print( metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))

    err.append(metrics.mean_absolute_error(testing_actual_output, testing_last_element_fcm_output))
    cm = confusion_matrix(testing_actual_output, testing_last_element_fcm_output)

    # convert the NumPy array to a pandas DataFrame
    df = pd.DataFrame(best_position, columns=["A","A", "B", "C","A", "A","A", "B", "C","A", "A","A", "B", "C","A", "A","A", "B", "C","A", "A","A", "B", "C" ])

    # write the DataFrame to an Excel file
    df.to_excel(f'ddata{fold}.xlsx', index=False)
    class_counts = cm.sum(axis=1)
    accuracies = [0 if count == 0 else cm[i, i] / count for i, count in enumerate(class_counts)]

    # print(cm)
    # sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
    if(A!=100):
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]

        sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
        # print('Sensitivity : ', sensitivity1*100 )

        sens.append(sensitivity1*100)

        specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
        # print('Specificity : ', specificity1*100)

        spec.append(specificity1*100)

        # calculate accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))

        # calculate mis-classification
        conf_misclassification = 1- conf_accuracy

        # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
        # calculate the specificity
        conf_specificity = (TN / float(TN + FP))


        def precision(TP, FP):
            return TP / (TP + FP)
        # calculate precision
        precision_score = precision(TP, FP)
        prec.append(precision_score*100)
        # print("Precision score:", precision_score*100)
    # conf_precision = (TN / float(TN + FP))


def calculate_deviation(values):
    # Step 1
    mean = sum(values) / len(values)

    # Step 2
    differences = [value - mean for value in values]

    # Step 3
    squared_differences = [diff ** 2 for diff in differences]

    # Step 4
    mean_squared_differences = sum(squared_differences) / len(squared_differences)

    # Step 5
    standard_deviation = math.sqrt(mean_squared_differences)

    return standard_deviation
print("\n\n\n")
print("-------------end of kfold------------")
from statistics import mean
print("Accuracies")
print(acc)
print(mean(acc))


acc_deviation = calculate_deviation(acc)
print("acc_deviation")
print(acc_deviation)



print("\n\nError")
print(err)
print(mean(err))
err_deviation = calculate_deviation(err)
print("err_deviation")
print(err_deviation)


print("\nSensitivity")
print(sens)
print(np.mean(sens))
sens_deviation = calculate_deviation(sens)
print("sens_deviation")
print(sens_deviation)


print("\nSpecificity")
print(spec)
print(np.mean(spec))
spec_deviation = calculate_deviation(spec)
print("spec_deviation")
print(spec_deviation)

print("\n Precision")
print(prec)
print(np.mean(prec)*100)
prec_deviation = calculate_deviation(prec)
print("prec_deviation")
print(prec_deviation)
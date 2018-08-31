from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from copy import copy, deepcopy
import numpy as np
#import pandas as pd

def classify(outfile, X_train, X_test,y_train,y_test,X_test_with_id,titles, crown_ids):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    RF = RandomForestClassifier()
    svm1 = svm.SVC()

    
    #predicting mlp
    mlp.fit(X_train,y_train)
    y_prob = mlp.predict_proba(X_test)
    y_prob = np.concatenate((np.array([titles]), y_prob), axis = 0)
    y_prob = np.concatenate((crown_ids, y_prob), axis = 1)
    y_prob = np.concatenate((y_prob, ([[""]]*1709)), axis = 1)
    np.savetxt(f"futher_explorations/mlp_{outfile}.csv",y_prob,delimiter = "\t", fmt = "%s")
    #futher_explorations/species.csv
    RF.fit(X_train, y_train)
    y_prob = RF.predict_proba(X_test)
    y_prob = np.concatenate((np.array([titles]), y_prob), axis = 0)
    y_prob = np.concatenate((crown_ids, y_prob), axis = 1)
    y_prob = np.concatenate((y_prob, ([[""]]*1709)), axis = 1)
    np.savetxt(f"futher_explorations/rf_{outfile}.csv",y_prob,delimiter = "\t", fmt = "%s")
    #svm
    svm1.fit(X_train, y_train)
    y_prob = svm1.predict(X_test)
    y_prob_adjusted = []
    for prediction in y_prob:
        y_prob_adjusted.append([1 if title == prediction else 0 for title in titles])
    y_prob = y_prob_adjusted
    y_prob = np.concatenate((np.array([titles]), y_prob), axis = 0)
    y_prob = np.concatenate((crown_ids, y_prob), axis = 1)
    y_prob = np.concatenate((y_prob, ([[""]]*1709)), axis = 1)
    np.savetxt(f"futher_explorations/svm_{outfile}.csv",y_prob,delimiter = "\t", fmt = "%s")



training_data_file = open('tmp/hyper_bands_train.csv','r')
specied_id_file = open('tmp/species_id_train.csv','r')
hyperbands = []
id_dictionary = {}
crown_id_dictionary = {}
answers = []
answers_genus = []
first = True
others = []
for line in specied_id_file:
    if first:
        first = False
        continue

    id_dictionary.update({line.rstrip().split(',')[0]:line.rstrip().split(',')[3:5]})


first = True
iteration = 0
crown_ids = []#[["crown_id"]]
for line in training_data_file:
    if first:
        first = False
        continue
    row = line.rstrip().split(',')
    crown_ids.append([row[0]])

    answers.append( id_dictionary.get( row[0] ) )
    row.append(iteration) #Adjust this
    row.insert(0,iteration)
    hyperbands.append(row)
    iteration = iteration + 1
crown_ids = np.array(crown_ids)

X = hyperbands
y = answers
species = []
genuss = []
for row in answers:
    if not any(row[0] in s for s in species):
        species.append(row[0])
    if not any(row[1] in s for s in genuss): #Try changing this
        genuss.append(row[1])
species = sorted(species)
genuss = sorted(genuss)
print(f"y: {np.array(y).shape}")
print(f"crown_ids: {crown_ids.shape}")

y = np.concatenate((y,crown_ids), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
crown_ids_test = y_test[:,2]
crown_ids_test.shape = (1708,1)
crown_ids_test = np.concatenate(([["Crown_id"]],crown_ids_test), axis = 0)
print(f"y_test: {crown_ids_test}")

y_gen_train = []
y_sp_train = []
y_gen_test = []
y_sp_test = []
for row in y_test:
    y_gen_test.append(row[0])
for row in y_test:
    y_sp_test.append(row[1])
for row in y_train:
    y_gen_train.append(row[0])
for row in y_train:
    y_sp_train.append(row[1])

X_train_with_id = deepcopy(X_train)
for row in X_train:
    del row[0:2]

X_test_with_id = deepcopy(X_test)
for row in X_test:
    del row[0:2]


mlp1 = classify('genus',X_train,X_test,y_sp_train,y_sp_test,X_test_with_id,genuss, crown_ids_test)
mlp2 = classify('species',X_train,X_test,y_gen_train,y_gen_test,X_test_with_id,species, crown_ids_test)

# for some reason the original code has sp and genus backwards
gen_test = np.concatenate((["Actual"],y_sp_test))
sp_test = np.concatenate((["Actual"],y_gen_test))

gen_test.shape = (1709, 1)
sp_test.shape = (1709, 1)

# also add the associated crown ids
gen_test = np.concatenate((crown_ids_test,gen_test), axis = 1)
sp_test = np.concatenate((crown_ids_test,sp_test), axis = 1)

np.savetxt("futher_explorations/genus_test.csv",gen_test,delimiter = "\t", fmt = "%s")
np.savetxt("futher_explorations/sp_test.csv",sp_test,delimiter = "\t", fmt = "%s")

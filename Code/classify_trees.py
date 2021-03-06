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

def classify(outfile, X_train, X_test,y_train,y_test,X_test_with_id):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    RF = RandomForestClassifier()
    svm1 = svm.SVC()

    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    #y_prob = mlp.predict_proba(X_test)
#    RF.fit(X_train, y_train)
#    predictions = RF.predict(X_test)
#    y_prob = RF.predict_proba(X_test)
#    y_prob = RF.predict_log_proba(X_test)

#    print(y_prob)
#    print(y_prob.shape)
#    y_prob.tofile(outfile, "\t", format="%s")
#    np.savetxt(outfile, y_prob, delimiter='\t')

#    X_Val = np.concatenate((X_train,X_test),axis=0)
#    Y_Val = np.concatenate((y_train,y_test),axis=0)
     ## The cross_val_score method was used to get estimates in predictions for graphing
#    print("MultiLayer Perceptron: ")
#    scores = cross_val_score(mlp,X_Val, Y_Val)
#    print(scores.mean())

#    print("Random Forest: ")
#    scores = cross_val_score(RF,X_Val, Y_Val)
#    print(scores.mean())

#    print("support vector machine: ")
#    scores = cross_val_score(svm1,X_Val, Y_Val)
#    print(scores.mean())

    return mlp


def predict(mlp1,mlp2,species,genuss):
    #prepate the date
    input_file = open('tmp/hyper_bands_test.csv','r')
    hyperbands = []
    iteration = 0
    first = True
    for line in input_file:
        if first:
            first = False
            continue
        hyperbands.append(line.rstrip().split(','))
        hyperbands[len(hyperbands)-1].insert(0,iteration)
        iteration = iteration + 1
    X = []
    pixel_and_crown_id = []
    for row in hyperbands:
        X.append([float(i) for i in row[2:]])
        pixel_and_crown_id.append(row[:2])

    predictions1 = mlp1.predict(X)
    predictions2 = mlp2.predict(X)
    outfile = open('gen.csv','w')
    outfile2 = open('species.csv','w')
    probs1 = mlp1.predict_proba(X)
    probs2 = mlp2.predict_proba(X)
    line = 'crown_id\t'
    for s in genuss:
        line = line + s + '\t'
    outfile.write(line + '\n')
    for i in range(0,len(predictions1)):
        line =  str(pixel_and_crown_id[i][1]) + '\t'
        for item in probs1[i]:
            line = line + str(item) + '\t'
        outfile.write(line + '\n')
    line = 'crown_id\t'
    for s in species:
        line = line + s + '\t'
    outfile2.write(line + '\n')
    for i in range(0,len(predictions2)):
        line = str(pixel_and_crown_id[i][1]) + '\t'
        for item in probs2[i]:
            line = line + str(item) + '\t'
        outfile2.write(line + '\n')


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
for line in training_data_file:
    if first:
        first = False
        continue
    row = line.rstrip().split(',')

    answers.append( id_dictionary.get( row[0] ) )
    row.append(iteration) #Adjust this
    row.insert(0,iteration)
    hyperbands.append(row)
    iteration = iteration + 1


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
X_train, X_test, y_train, y_test = train_test_split(X, y)
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


mlp1 = classify('graphs/species_rf.csv',X_train,X_test,y_sp_train,y_sp_test,X_test_with_id)
mlp2 = classify('graphs/genus_rf.csv',X_train,X_test,y_gen_train,y_gen_test,X_test_with_id)

predict(mlp1,mlp2,species,genuss)


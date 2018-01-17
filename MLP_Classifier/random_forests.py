#This is the MLP Classifier I got up and running, it's precision is ~95%
#I hope this servers as a good example for any one that needs one
# -Michael
#this is a link to the tutorial I followed to figure this out
#https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#these files need to be in the same dirrectory as this script, or you need to change the file paths
#also I removed the last column from hyper_bands_train.csv prior to running this
#my reasoning for this is not all the reads have information in this column which was causing error with the analysis
training_data_file = open('hyper_bands_train.csv','r')
specied_id_file = open('species_id_train.csv','r')

hyperbands = [] #will contain all the data for the various crowns

id_dictionary = {}

answers = [] #will contain the two letter species abbreviation
first = True
#this reads in the input file, skipping the first line and adding the crown ID and Genus ID to a dictionary
for line in specied_id_file:
    if first:
        first = False
        continue
    id_dictionary.update({line.rstrip().split(',')[0]:line.rstrip().split(',')[4]})

first = True
#imports the actual data into a 2d Array, skipping the first row.
#simitatiously creates an array called answers of the that contains the corresponding Genus Id for ever row read in
for line in training_data_file:
    if first:
        first = False
        continue
    row = line.rstrip().split(',')
    hyperbands.append(row[2:])
    thing = id_dictionary.get( row[0] )
    answers.append( id_dictionary.get( row[0] ) )

X = hyperbands
y = answers

#splits the training data into two sets
#training sets contain 70% of the reads
#testing sets contain the remaining 30%
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()

#scales and normalized the data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=245, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())

clf = ExtraTreesClassifier(n_estimators=245, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean()) 


#train your object by passing in the reads and corresponding solutions
#mlp.fit(X_train,y_train)

#predictions = mlp.predict(X_test)

#this is what the program predicts each of the testing reads to be
#print(predictions)

#this is a confision matrix, I cannot give a good explaination of what it means
#but is has to do with the result accuracy, google it for a better explaination
#print(confusion_matrix(y_test,predictions))

#this is the important resport that tells of the programs precision
#print(classification_report(y_test,predictions))

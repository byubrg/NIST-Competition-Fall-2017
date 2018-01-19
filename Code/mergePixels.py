import sys
from operator import itemgetter

inFile = sys.argv[1]
outFile = sys.argv[2]
type = sys.argv[3]

genusList = []
headerList = []
indecisOrder = []

def crownResultsToString(curList) :
    result = ""
    for i in range(len(indecisOrder) - 1) :
        result = result + str(curList[0])+ "," + str(headerList[i + 1]) + "," + str(curList[indecisOrder[i + 1]]) + "\n"
    return(result)

with open(inFile, "r") as inF :
    headerList = inF.readline().rstrip().split('\t')
    for line in inF :
        lineList = line.strip('\n').split('\t')[:-1]
        lineList = [float(value) for value in lineList]
        lineList[0] = int(lineList[0])
        genusList.append(lineList)
headerDict = {}
for i in range(len(headerList)) :
    headerDict[headerList[i]] = i


headerList[1:] = sorted(headerList[1:])


for i in range(len(headerList)) :
   indecisOrder.append(headerDict[headerList[i]])

genusList = sorted(genusList, key=itemgetter(0))


curCrown = 0
first = True
curList = []
count = 0
innerListTemp = []

with open(outFile, "w") as oF :
    oF.write("Crown_id," + type + ",Probability\n")
    for innerList in genusList:

        innerListTemp = innerList
        count = count + 1
        if first :
            first = False
            curList = innerList
            curCrown = innerList[0]
            count = 0
        elif innerList[0] == curCrown :
            for i in range(len(innerList) - 1) :
                curList[i + 1] = curList[i + 1] + innerList[i + 1]
        else :
            for i in range(len(innerList) - 1) :
                curList[i + 1] = curList[i + 1] / count
            count = 0 
            oF.write(crownResultsToString(curList))
            curList = innerList
            curCrown = innerList[0]
    count = count + 1
    for i in range(len(innerList) - 1) :
        curList[i + 1] = curList[i + 1] / count
    oF.write(crownResultsToString(curList))


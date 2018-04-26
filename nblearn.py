import numpy as np, sys, json, math


# write human readable model for inspection
def writeHumanReadable(wordpos, prior, likelihood):
    outputFile = open('nbmodel.txt', "w")
    n = len(wordpos)
    outputFile.write("List of words with row num:\n")

    for key in wordpos:
        outputFile.write("{" + str(key) + ":" + str(wordpos[key]) + "} ,")

    outputFile.write("\n\nPrior probabilites of True, Fake, Pos, Neg:-\n\n")
    np.savetxt(outputFile, prior)

    outputFile.write("\n\nLikelihoods of words in list of words against 4 classes:-\n\n")
    np.savetxt(outputFile, likelihood)

    outputFile.close()


# write output files
def writeOutput(wordpos, prior, likelihood):
    with open('wordpos.txt', 'w') as inputFile:
        inputFile.write(json.dumps(wordpos))
    np.savetxt('prior.txt', prior)
    np.savetxt('likelihood.txt', likelihood)

    writeHumanReadable(wordpos, prior, likelihood)


# create a dict of unique words along with positions
def createVocab(fileLines, stopwords):
    wordPos = dict()
    v = 0
    for line in fileLines:
        words = line.split(" ")[3:]
        for word in words:
            if word=="":
                continue
            if word in stopwords:
                continue
            if word not in wordPos:
                wordPos[word] = v
                v += 1

    return wordPos, v


# increment wordCounts for each of the 4 classes
def incrementClassCounts(line, prior, counts):
    c1 = line.split(" ")[1]
    c2 = line.split(" ")[2]
    # stores class number from 1..4
    pos1 = 0
    pos2 = 2
    l = len(line.split(" ")[3:])
    if c1 == "true":
        prior[0] += 1
        counts[0] += l
    else:
        prior[1] += 1
        counts[1] += l
        pos1 = 1

    if c2 == "pos":
        prior[2] += 1
        counts[2] += l
    else:
        prior[3] += 1
        counts[3] += l
        pos2 = 3
    return prior, pos1, pos2, counts


# get the input set of lines in lower case
def getInput():
    inputFile = open("coding-2-data-corpus/train-labeled.txt", "r").read().lower()
    #inputFile = open(sys.argv[1], "r").read().lower()
    #inputFile = open("temp.txt", "r").read().lower()

    puncList = [".", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\""]
    for punc in puncList:
        inputFile = inputFile.replace(punc, '')

    fileLines = inputFile.split("\n")

    # remove last new line
    if (fileLines[len(fileLines) - 1]) == "":
        fileLines = fileLines[:-1]

    # number of docs in D
    nDoc = len(fileLines)

    return fileLines, nDoc


# compute prior prob for each class and word likelihoods
def computeProb(fileLines, wordPos, nDoc, v, stopwords):
    likelihood = np.ones((v, 4))
    prior = np.zeros(4)
    counts = np.zeros(4)

    for line in fileLines:

        prior, pos1, pos2, counts = incrementClassCounts(line, prior, counts)
        words = line.split(" ")[3:]

        for word in words:
            if word=="":
                continue
            if word in stopwords:
                continue
            likelihood[wordPos[word]][pos1] += 1
            likelihood[wordPos[word]][pos2] += 1

    for i in xrange(v):
        for j in xrange(4):
            likelihood[i][j] /= (counts[j] + v)

    likelihood = np.log(likelihood)
    prior = np.log(prior / nDoc)

    '''print prior
    for key, value in sorted(wordPos.iteritems(), key=lambda (k, v): (v, k)):
        print "%s: %s %s" % (key, value,str(likelihood[value]))'''

    return wordPos, prior, likelihood


def main():
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                 'now']

    fileLines, nDoc = getInput()
    wordPos, v = createVocab(fileLines, stopwords)
    wordPos, prior, likelihood = computeProb(fileLines, wordPos, nDoc, v, stopwords)
    writeOutput(wordPos, prior, likelihood)


main()
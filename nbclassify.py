import numpy as np, json, sys, math


def getInput():
    wordtext = open("wordpos.txt", "r").read()
    wordpos = json.loads(wordtext)

    prior = np.loadtxt("prior.txt")
    likelihood = np.loadtxt("likelihood.txt")

    return wordpos, prior, likelihood


def getLineScore(line, wordpos, likelihood, classNum):
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
    res = 0.0
    for word in line:
        word = word.lower()
        if word in wordpos:
            if word in stopwords:
                continue
            # remove high frequency stop words
            # if likelihood[wordpos[word]][classNum] > 0.1:
            #   continue
            res += likelihood[wordpos[word]][classNum]

    return res


# compute score for each pair of classes and output the higher
def nbclassify(inputFile, wordpos, prior, likelihood):
    outputFile = open("nboutput.txt", "w")

    inputFile = inputFile.read()

    puncList = [".", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\""]
    for punc in puncList:
        inputFile = inputFile.replace(punc, '')

    fileLines = inputFile.split("\n")
    for line in fileLines:
        if line == "":
            continue

        key = line.split(" ")[0]
        text = line.split(" ")[1:]

        class1Name = "True"
        class2Name = "Pos"

        if (prior[1] + getLineScore(text, wordpos, likelihood, 1)) > (
                prior[0] + getLineScore(text, wordpos, likelihood, 0)):
            class1Name = "Fake"

        if (prior[3] + getLineScore(text, wordpos, likelihood, 3)) > (
                prior[2] + getLineScore(text, wordpos, likelihood, 2)):
            class2Name = "Neg"

        outputFile.write(key + " " + class1Name + " " + class2Name + "\n")

    outputFile.close()


def main():
    wordpos, prior, likelihood = getInput()
    #inputFile = open("coding-2-data-corpus/dev-text.txt", "r")
    inputFile = open(sys.argv[1], "r")
    nbclassify(inputFile, wordpos, prior, likelihood)
    inputFile.close()


main()
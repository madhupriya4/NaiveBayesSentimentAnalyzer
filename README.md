### Prediction of True/Fake and Pos/Neg labels by building two binary Naive Bayes classifiers on the training data

### Data

The uncompressed archive has the following files:

One file train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). 

The first 3 tokens in each line are:

a unique 7-character alphanumeric identifier

a label True or Fake

a label Pos or Neg


These are followed by the text of the review.

One file dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).

One file dev-key.txt with the corresponding labels for the development data, to serve as an answer key.


### Programs

nblearn.py will learn a naive Bayes model from the training data, and nbclassify.py will use the model to classify new data. 

The learning program will be invoked in the following way:

> python nblearn.py /path/to/input

The argument is a single file containing the training data; the program will learn a naive Bayes model, and write the model parameters to a file called nbmodel.txt. 

The classification program will be invoked in the following way:

> python nbclassify.py /path/to/input

The argument is a single file containing the test data file; the program will read the parameters of a naive Bayes model from the file nbmodel.txt, classify each entry in the test data, and write the results to a text file called nboutput.txt in the same format as the answer key.
# NaiveBayesSentimentAnalyzer

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
#from nltk.corpus import movie_reviews

from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
   
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

all_words = []
documents = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
    
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(20))
#print(all_words["great"])

word_features = list(all_words.keys())[:5000]

save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets = open("featuresets5k.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print(len(featuresets))

#positive data example:
train_set = featuresets[:10000]
test_set = featuresets[10000:]

#negative data example:
#train_set = featuresets[100:]
#test_set = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)

#classifier_f = open("naivebayes.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()

print("Original Naive Bayes Accuracy Percentage: ", (nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MNB_classifier Accuracy Percentage: ", (nltk.classify.accuracy(MNB_classifier, test_set))*100)

save_classifier = open("MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, test_set))*100)


save_classifier = open("BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

save_classifier = open("LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(train_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, test_set)*100)

save_classifier = open("SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

save_classifier = open("LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(train_set)
#print("NuSVC Accuracy Percentage: ", (nltk.classify.accuracy(NuSVC, test_set))*100)

#save_classifier = open("NuSVC_classifier5k.pickle","wb")
#pickle.dump(NuSVC_classifier, save_classifier)
#save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier)
print("voted_classifier accuracy percentage:",(nltk.classify.accuracy(voted_classifier, test_set))*100 )

#print("Classification:", voted_classifier.classify(test_set[0][0]), "Confidence:", voted_classifier.confidence(test_set[0][0])*100)
#print("Classification:", voted_classifier.classify(test_set[1][0]), "Confidence:", voted_classifier.confidence(test_set[1][0])*100)
#print("Classification:", voted_classifier.classify(test_set[2][0]), "Confidence:", voted_classifier.confidence(test_set[2][0])*100)
#print("Classification:", voted_classifier.classify(test_set[3][0]), "Confidence:", voted_classifier.confidence(test_set[3][0])*100)
#print("Classification:", voted_classifier.classify(test_set[4][0]), "Confidence:", voted_classifier.confidence(test_set[4][0])*100)
#print("Classification:", voted_classifier.classify(test_set[5][0]), "Confidence:", voted_classifier.confidence(test_set[5][0])*100)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats)

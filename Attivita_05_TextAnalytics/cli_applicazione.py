# !pip install happiestfuntokenizing xgboost spacy
# !python -m spacy download en_core_web_s


from pymongo import MongoClient

client = MongoClient('localhost', 27017)
DATABASE = client.test


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import nltk, random
from nltk.sentiment.util import mark_negation, extract_unigram_feats
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.classify import NaiveBayesClassifier


NUM_RUN = int(input("Quanti run eseguire? "))
REVIEWS_PER_CLASSE = 1000
REVIEWS_PER_CLASSE_TEST = 100

def getReviewData(lista_overall,  field="reviewText", funzione_tokenize=None):
    """ 
    Questa funzione ritorna una lista di tuple (rec, classe) dove rec è la recensione e classe è la classificazione (es. 1.0 o 5.0).
        - lista_overall è la lista delle classi (campo overall delle review)
        - field è il campo della review che si vuole ottenere
        - funzione_tokenize è la funzione usata per tokenizzare il testo della review. Se non dato, la lista ritornata conterra' il contenuto del campo field
    """
    reviews = []
    for overall in lista_overall:
        # for review in DATABASE.reviews.find({"overall": overall}).limit(REVIEWS_PER_CLASSE):
        for review in DATABASE.reviews.aggregate([
            { '$match' :{"overall": overall}},  
            { '$match': { '$expr': { '$lt': [0.5, {'$rand': {} } ] } } },   # Prendo le reviews in modo random ogni volta
            { '$limit' : REVIEWS_PER_CLASSE}]):
            if funzione_tokenize == None:
                reviews.append((review[field],str(overall))) 
            else:
                reviews.append(([str(i).lower() for i in funzione_tokenize(review[field])],str(overall))) 
            
    random.shuffle(reviews) 
    return reviews 

def get_classifier_and_mean_accuracy(lista_overall, funzione_filtra_reviews=lambda x:x, feat_extractor=None,  **kwargs):
    """
    Questa funzione ritorna il classifier e la relativa accuracy media.  
        - lista_overall è la lista delle classi (campo overall delle review)
        - funzione_filtra_reviews è una funzione che viene chiamata per cambiare le reviews da esaminare. 
          funzione_filtra_reviews prende in ingresso le reviews estratte (lista di tuple (rec, classe)). 
          funzione_filtra_reviews può poi, ad esempio, selezionare solo alcune parole (es. solo gli aggettivi) dai testi delle recensioni.
          Infine, essa deve ritornare le reviews modificate.
        - feat_extractor è una funzione che viene usata nell'add_feat_extractor del sentiment analyzer. 
          Se omessa, viene usata extract_unigram_feats. 
          Altrimenti, feat_extractor deve essere una funzione che prende in input i soli documenti
        - i restanti parametri opzionali saranno dati in input a getReviewData
    """

    # Run dei test e accuracy
    accuracy = []

    for i in range(NUM_RUN):
        # Prendo le reviews
        reviews = getReviewData(lista_overall, **kwargs)
        reviews = funzione_filtra_reviews(reviews)
        
        # Sentiment analyzer e applicazione delle feature
        sentimAnalyzer = SentimentAnalyzer() 

        if feat_extractor == None:
            allWordsNeg = sentimAnalyzer.all_words([mark_negation(doc) for doc in reviews])    
            unigramFeats = sentimAnalyzer.unigram_word_feats(allWordsNeg, min_freq=4)   
            sentimAnalyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigramFeats)
        else:
            sentimAnalyzer.add_feat_extractor(feat_extractor)

        featureSets = sentimAnalyzer.apply_features(reviews)
        trainSet, testSet = featureSets[REVIEWS_PER_CLASSE_TEST:], featureSets[:REVIEWS_PER_CLASSE_TEST]

        # Classificatore e accuracy
        with HiddenPrints(): 
            classifier = sentimAnalyzer.train(NaiveBayesClassifier.train, trainSet)
            evalu = sentimAnalyzer.evaluate(testSet)

        accuracy.append(evalu["Accuracy"])

    return classifier, round(sum(accuracy)/len(accuracy), 4)

print("--------------------------\nVersione 1 (originale): Word tokenizer di nltk con due classi """)

classifier1, accuracy1 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_tokenize=nltk.word_tokenize)
print("Accuracy:", accuracy1)

dizionario_plot = {}   # Dizionario usato alla fine per stampare i risultati graficamente
dizionario_plot["1.Originale"] = accuracy1


classifier1.show_most_informative_features()

print("--------------------------\nVersione 2: Sentiment tokenizer di Christopher Potts con due classi""")


from happiestfuntokenizing.happiestfuntokenizing import Tokenizer
classifier2, accuracy2 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_tokenize=Tokenizer().tokenize)
print("Accuracy:", accuracy2)
dizionario_plot["2.ChrisPott"] = accuracy2


classifier2.show_most_informative_features()

print("--------------------------\nVersione 3: Classificazione a cinque classi""")


classifier3, accuracy3 = get_classifier_and_mean_accuracy([float(i) for i in range(1,6)], funzione_tokenize=nltk.word_tokenize)
print("Accuracy:", accuracy3)
dizionario_plot["3.Cinque classi"] = accuracy3


classifier3.show_most_informative_features()

print('--------------------------\nVersione 4: Uso del field "summary"')


classifier4, accuracy4 = get_classifier_and_mean_accuracy([1.0, 5.0], field="summary", funzione_tokenize=nltk.word_tokenize)
print("Accuracy:", accuracy4)
dizionario_plot["4.Summary"] = accuracy4

classifier4.show_most_informative_features()


print("--------------------------\nVersione 5: uso dei soli aggettivi""")


from nltk.corpus import wordnet as wn

def filtra_solo_aggettivi(reviews):
    new_reviews = []
    for (lista_parole,classe) in reviews:
        solo_aggettivi = []
        for par in lista_parole:
            tmp = wn.synsets(par)
            if len(tmp) > 0 and tmp[0].pos() == "a":
                solo_aggettivi.append(par)
        if solo_aggettivi != []:
            new_reviews.append((solo_aggettivi,classe)) 
    return new_reviews

classifier5, accuracy5 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_tokenize=nltk.word_tokenize, funzione_filtra_reviews=filtra_solo_aggettivi)
print("Accuracy:", accuracy5)
dizionario_plot["5.Solo aggettivi"] = accuracy5

classifier5.show_most_informative_features()

print("--------------------------\nVersione 6: uso della word frequency ")

def conta_parole(words):
    wfreq=[words.count(w) for w in words]
    return dict(zip(words,wfreq))

classifier6, accuracy6 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_tokenize=nltk.word_tokenize, feat_extractor=conta_parole)
print("Accuracy:", accuracy6)
dizionario_plot["6.Frequency word"] = accuracy6

classifier6.show_most_informative_features()


print("--------------------------\nVersione 7: WordNetLemmatizer")


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

stop_words = sorted(stopwords.words('english'))
wnl = WordNetLemmatizer()

def filtra_non_stop_words(reviews):
    reviews = [([wnl.lemmatize(parola) for parola in rec if parola not in stop_words], classe) for (rec,classe) in reviews]
    return reviews
    
classifier7, accuracy7 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_tokenize=nltk.word_tokenize,  funzione_filtra_reviews=filtra_non_stop_words, feat_extractor=conta_parole)
print("Accuracy:", accuracy7)
dizionario_plot["7.WordNetLemmatizer"] = accuracy7

classifier7.show_most_informative_features()

print("--------------------------\nVersione 8: Spacy")

import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_lemmatizer(reviews):    
    new_reviews = []
    for (testo, classe) in reviews:
        doc = nlp(testo)
        lemmas = [token.lemma_ for token in doc if token.lemma_ not in stop_words] 
        new_reviews.append((lemmas, classe))    
    return new_reviews
    
classifier8, accuracy8 = get_classifier_and_mean_accuracy([1.0, 5.0],funzione_filtra_reviews=spacy_lemmatizer, feat_extractor=conta_parole)
print("Accuracy:", accuracy8)
dizionario_plot["8.Spacy"] = accuracy8

classifier8.show_most_informative_features()


print("--------------------------\nVersione  9: lemmatize e opinion_lexicon")

from string import punctuation
from nltk.corpus import opinion_lexicon

nlp = spacy.load("en_core_web_sm")

neg_words = sorted(opinion_lexicon.negative())
pos_word = sorted(opinion_lexicon.positive())
pos_tag = {'VERB','NOUN','ADV', 'ADJ'}  

def get_keywords_in_text(text):
    for punteggiatura in punctuation:
        text = text.replace(punteggiatura, " ", -1)   
        
    doc = nlp(text.lower()) 
    result = [token.lemma_ for token in doc if (not token.text in stop_words ) and (token.pos_ in pos_tag)]
    result = [token for token in result if (token in neg_words or token in pos_word)]
    return result

def seleziona_keyword(reviews):
    # reviews = [(get_keyword_in_text(rec), classe) for (rec, classe) in reviews]
    for i in range(len(reviews)):
        # print(i, end=" ")
        reviews[i] = (get_keywords_in_text(reviews[i][0]), reviews[i][1])
    return reviews

classifier9, accuracy9 = get_classifier_and_mean_accuracy([1.0, 5.0], funzione_filtra_reviews=seleziona_keyword, feat_extractor=conta_parole) 
print("Accuracy:", accuracy9)
dizionario_plot["9.lemmatize,opinion_lexicon"] = accuracy9

classifier9.show_most_informative_features()



print("--------------------------\nVersione 10: Classificatore Support Vector Machines (SVM)")

def get_train_test_data_target(reviews):
    """
    Questa funzione ritorna: train_data, train_target, test_data e test_target delle reviews in input.
        - train_data, test_data sono i "documenti" del trainset e testset rispettivamente
        - train_target, test_target sono le classi del trainset e testset rispettivamente
    """
    classi = list(set(list({x[1] for x in reviews})))
    testSet= reviews[:REVIEWS_PER_CLASSE]  
    trainSet = reviews[REVIEWS_PER_CLASSE:]

    # classi_count = {str(c):0 for c in classi}
    # for (feat, classe) in trainSet:
    #     classi_count[classe] +=1
    # print("trainset:", classi_count)

    # classi_count = {str(c):0 for c in classi}
    # for (feat, classe) in testSet:
    #     classi_count[classe] +=1
    # print("testSet:", classi_count)

    
    train_data, train_target = list(map(list, zip(*trainSet)))
    test_data, test_target = list(map(list, zip(*testSet)))

    return train_data, train_target, test_data, test_target


from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline

# Creo il train e test set
reviews10 = getReviewData([1.0, 5.0])
train_data, train_target, test_data, test_target = get_train_test_data_target(reviews10)

# Classificatore
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', svm.SVC()),])
text_clf.fit(train_data, train_target)
predicted = text_clf.predict(test_data)

# Cross-validation
cv = ShuffleSplit(n_splits=NUM_RUN, test_size=0.2, random_state=0)
scores = cross_val_score(text_clf, test_data, test_target, cv=cv)
accuracy10 = round(sum(scores)/len(scores), 4)
print("Accuracy:", accuracy10)
dizionario_plot["10.SVM"] = accuracy10



print("--------------------------\nVersione 11: Hyperlane")

from sklearn.linear_model import SGDClassifier

# Creo il train e test set
reviews11 = getReviewData([1.0, 5.0])
train_data, train_target, test_data, test_target = get_train_test_data_target(reviews11)

# Classificatore
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),])
text_clf.fit(train_data, train_target)
predicted = text_clf.predict(test_data)

# Cross-validation
cv = ShuffleSplit(n_splits=NUM_RUN, test_size=0.2, random_state=0)
scores = cross_val_score(text_clf, test_data, test_target, cv=cv)
accuracy11 = round(sum(scores)/len(scores), 4)
print("Accuracy:", accuracy11)
dizionario_plot["11.Hyperlane"] = accuracy11


print("--------------------------\nVersione 12: Grid-search")

from sklearn.model_selection import GridSearchCV

reviews12 = getReviewData([1.0, 5.0])
train_data, train_target, test_data, test_target = get_train_test_data_target(reviews12)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf.fit(train_data, train_target)

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
cv = ShuffleSplit(n_splits=NUM_RUN, test_size=0.2, random_state=0)

gs_clf = GridSearchCV(text_clf, parameters, cv=cv, n_jobs=-1)
gs_clf.fit(train_data, train_target)
predicted = gs_clf.predict(test_data)

scores = cross_val_score(text_clf, test_data, test_target, cv=cv)
accuracy12 = round(sum(scores)/len(scores), 4)

print("Accuracy: ", accuracy12)
dizionario_plot["12.Grid-search"] = accuracy12

print("--------------------------\nVersione 13: Uso di XGBoost")

import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score

accuracy13 = []
for _ in range(NUM_RUN):
    reviews13 = getReviewData([1.0, 5.0])
    train_data, train_target, test_data, test_target = get_train_test_data_target(reviews13)

    cv = CountVectorizer(binary = True)
    cv.fit(train_data, train_target)
    train_transform = cv.transform(train_data)
    test_transform = cv.transform(test_data)

    # Trasformo i target del trainset e testset in un formato adatto a xgboost
    train_target = [0 if target=="1.0" else 1 for target in train_target]
    test_target = [0 if target=="1.0" else 1 for target in test_target]

    # Training and Predicting
    train_matrix = xgb.DMatrix(train_transform, train_target)      
    test_matrix = xgb.DMatrix(test_transform, test_target)

    param = {'objective': 'binary:hinge'}       # Mette in  predicted_test solo valori 0 o 1 
    classifier13 = xgb.train(param, train_matrix, num_boost_round = 30)
    predicted_test = classifier13.predict(test_matrix).tolist()

    # Accuracy
    accuracy13 += [accuracy_score(test_target, predicted_test)]

accuracy13 = round(sum(accuracy13)/len(accuracy13), 4)
print("Accuracy:",  accuracy13)
dizionario_plot["13.Xgboost"] = accuracy13

print("\n--------------------------\n")
print("Risultati e conclusioni")
print(f"{'Versione del classificatore':29}", "Accuracy")
[print(f"{nome_class:30}" + str(accuracy)) for (nome_class, accuracy) in sorted(dizionario_plot.items(), key=lambda x:x[1], reverse=True)]

import matplotlib.pyplot as plt

x_titoli , y_accuracy = list(dizionario_plot.keys()), list(dizionario_plot.values())
plt.rcParams["figure.figsize"] = (24,7)

plt.plot(x_titoli, y_accuracy)
for i in range(len(x_titoli)):
    if y_accuracy[i] == max(y_accuracy):
        plt.text(x_titoli[i], y_accuracy[i], str(y_accuracy[i]), color="red", weight='bold')
    else:
        plt.text(x_titoli[i], y_accuracy[i], str(y_accuracy[i]))

plt.title('Accuracy delle versioni')
plt.grid()
plt.show()

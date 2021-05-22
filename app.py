
from flask import Flask, request, render_template

import glob
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.metrics import classification_report

import nltk
from nltk.stem.snowball import SnowballStemmer
#from nltk import sent_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import networkx as nx


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from operator import itemgetter
import spacy
nlp = spacy.load("en_core_web_sm")


#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')



app=Flask(__name__,template_folder='templates')


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/redirect',methods=['POST','GET'])
def redirect():
    if request.method == 'POST':
        if request.form.get('Multi Doc'):
            return render_template('multi.html')
        elif request.form.get('Single Doc'):
            return render_template('single.html') 


################## MULTI - DOCUMENT #####################

#Defining pre-process function
def pre_process(doc):
    print("Pre Processing...")
    tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
    tokens= [w for w in tokens if w not in stopwords]
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stems = [stemmer.stem(t) for t in tokens]
    return stems


## called in summary_bullets after getting sub-topics
def summary_sents(sub_topic, sents):
    print("Summary Sentences...")
    sents = sent_tokenize(' '.join(sents))
    sents.append(sub_topic)

    sent_vectorizer = TfidfVectorizer(decode_error='replace', min_df=1, stop_words='english',
                                      use_idf=True, tokenizer=pre_process, ngram_range=(1, 3))

    sent_tfidf_matrix = sent_vectorizer.fit_transform(sents)
    # subtopic_tfidf = sent_tfidf_matrix.transform([sub_topic])
    sub_topic_similarity = cosine_similarity(sent_tfidf_matrix)
    top10_sents = sub_topic_similarity[-1][:-1].argsort()[:-11:-1]
    final_sents = []
    for i in top10_sents:
        final_sents.append(sents[i])
    return final_sents


## called in ext_summary to generate extrative summary of topic
def summarize(text):
    print("Summary...")
    sentences_token = sent_tokenize(text)

    # Feature Extraction
    vectorizer = CountVectorizer(min_df=1, decode_error='replace')
    sent_bow = vectorizer.fit_transform(sentences_token)
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    sent_tfidf = transformer.fit_transform(sent_bow)

    similarity_graph = sent_tfidf * sent_tfidf.T

    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    text_rank_graph = sorted(((scores[i], s) for i, s in enumerate(sentences_token)), reverse=True)
    #      print(scores)
    number_of_sents = int(0.4 * len(text_rank_graph))
    del text_rank_graph[number_of_sents:]
    summary = ' '.join(word for _, word in text_rank_graph)

    return summary


# tf-idf
def vectors(docs):
    print("Creating Vectors...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, decode_error='replace',
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=pre_process, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    terms = tfidf_vectorizer.get_feature_names()
    return tfidf_matrix, terms, tfidf_vectorizer

# print(tfidf_vectorizer)
# print(tfidf_matrix)
# print(terms)

#Clustering
def groups(tfidf_matrix):
    print('Clustering Documents...')
    corpus_similarity = cosine_similarity(tfidf_matrix)#Similarity
    km = KMeans(n_clusters=3)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters

#print(corpus_similarity)
#print(km)
#print(clusters)




# topic extraction
def topic_modelling(terms,tfidf_matrix):
    print('Modelling Topics...')
    lda = LatentDirichletAllocation(n_components=3, max_iter=200, learning_method='online',
                                    learning_offset=50., random_state=42)
    lda.fit(tfidf_matrix)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topics.append(" ".join([terms[i] for i in topic.argsort()[:-30 - 1:-1]]))
    return topics

# print(lda)
# print(topics)

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")




@app.route('/readData',methods=['POST','GET'])
def readData():
    # Read files
    # C:\Users\teja.raju\Desktop\College\Mini Projects and Projects\Project\BBC News Summary\News Articles\sport
    ip = [x for x in request.form.values()]
    path = ip[0]
    # path = input('Please Enter Directory Path: ')
    print('Reading files...')
    path = path + "/*.txt"
    files = glob.glob(path)
    docs = []
    for name in files:
        with open(name) as f:
            docs.append(f.read())

    # add title as headline
    fname = [fname[12:] for fname in files]
    fname = [re.sub(".txt", ". ", n) for n in fname]
    docs = list(map(str.__add__, fname, docs))

    # print(fname)
    # print(docs)
    # print(stemmer)

    # vectors
    tfidf_matrix, terms, tfidf_vectorizer = vectors(docs)

    # groups
    clusters = groups(tfidf_matrix)

    # topic_modelling
    topics = topic_modelling(terms, tfidf_matrix)

    # finding similar topic
    query = ip[1]
    topics.append(query)
    tfidf_topics_matrix = tfidf_vectorizer.fit_transform(topics)
    topic_similarity = cosine_similarity(tfidf_topics_matrix)
    topics = topics[:-1]

    # index of most similar cluster
    similar_clust = np.argmax(topic_similarity[3][:3])
    article_indices = [i for i, x in enumerate(clusters) if x == similar_clust]

    print('Building Extractive Summary...')

    # extractive summary
    ext_summary = []
    for i in article_indices:
        ext_summary.append(summarize(docs[i]))  ## calls summarize(text)

    # print(len(ext_summary))
    # exit()

    sub_topic = []
    print('Making Bullet Points...')
    sub_topic.append(query)
    summary_bullets = summary_sents(sub_topic[0], ext_summary)  ## calls summary_sents(subtopic, sents)
    # print("\n" + sub_topic[0] + "\n")

    out = ''
    for b in summary_bullets:
        out = out + '\n' + '\n * ' + b
    print(out)
    return render_template('multi.html', summary=out, heading=sub_topic[0])




#################### SINGLE - DOCUMENT #####################

@app.route('/textsummarization',methods=['POST', 'GET'])
def textsummarization():
    #sen = "Computer engineering is a relatively new field of engineering and is one of the fastest growing fields today. Computer engineering is one of today’s most technologically based jobs. The field of computer engineering combines the knowledge of electrical engineering and computer science to create advanced computer systems. Computer engineering involves the process of designing and manufacturing computer central processors, memory systems, central processing units, and of peripheral devices. Computer engineers work with CAD(computer aided design) programs and different computer languages so they can create and program computer systems. Computer engineers use today’s best technology to create tomorrow’s. Computer engineers require a high level of training and intelligence to be skilled at their job. A bachelors degree from a college or university with a good computer engineering program computer science program is necessary. Then once employed their usually is a on the job type of training program to learn the certain types of systems that will be designed and manufactured. Computer engineers major studies conventional electronic engineering, computer science and math in college. The electrical engineering knowledge that a computer engineer possesses allow for a understanding of the apparatus that goes into a computer so that they can be designed and built."
    s = [x for x in request.form.values()]
    sen=''
    for i in s:
        sen+=i
    sentences = list(sen.split('. '))
    stop_words = nltk.corpus.stopwords.words('english')
    out=''
    for idx, sentence in enumerate(textrank(sentences, stop_words)):
       out=out+''+''.join(sentence)+'. '
    return render_template('single.html',summary=out,heading="Summary")

def textrank(sentences, stopwords=None, top_n=5):
    S = build_similarity_matrix(sentences, stopwords)
    #print("SIMILARITY_MATRIX\n")
    #print(S)
    sentence_ranks = pagerank(S)
    #print("\n\n\nSENTENCE_RANKING\n")
    #print(sentence_ranks)
    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:int(top_n)])
    #print("\n\n\nRANKED_INDEXES\n")
    #print(ranked_sentence_indexes)
    summary = itemgetter(*selected_sentences)(sentences)
    #print("\n\n\nRanked_Summary\n")
    return summary

def build_similarity_matrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue

            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

    # normalize the matrix row-wise
    for idx in range(len(S)):
        S[idx] /= S[idx].sum()
    return S

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)  # coloumn matrix
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)  # (A^T).p
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def sentence_similarity(sent_1, sent_2, stopwords=None):
    # lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    sent_1 = nlp(sent_1)
    sent_2 = nlp(sent_2)
    sent1 = " ".join([token.lemma_ for token in sent_1])
    sent2 = " ".join([token.lemma_ for token in sent_2])

    # Removing Stop Word
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)


if __name__ == '__main__':
    app.run(debug=True)
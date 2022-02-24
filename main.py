from googleapiclient.discovery import build
import pprint as pp
import sys
import json
import math
import numpy as np
#from nltk.tokenize import word_tokenize
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
from autocorrect import Speller

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

'''
How to install nltk
follow these instructions: https://www.nltk.org/install.html 
type python in terminal 
import nltkimport 
nltk.download()
nltk.download('punkt')

then these instructions to install punkt https://github.com/gunthercox/ChatterBot/issues/930#issuecomment-322111087
crl d to get back to regular terminal 

pip install autocorrect
pip install -U scikit-learn
'''


def get_search_results(query, api_key, engine_id):
    # call api with query and return result
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=engine_id, ).execute()
    return res


def unwrap_json(search_results):
    top_10_results_json_dump = search_results['items']
    # print("top_10_results", top_10_results_json_dump)
    return top_10_results_json_dump


def relevant():
    relevant = input("Relevant (Y/N)?").upper()
    # add new key value pair to dict
    if relevant == "Y":
        return 1
    elif relevant == "N":
        return 0
    else:
        print("terminate")
        sys.exit()


def print_search_params(api_key, engine_key, query, precision):
    print("Parameters")
    print("Client key =", api_key)
    print("Engine key =", engine_key)
    print("Query      =", query)
    print("Precision  =", precision)
    print("Google Search Results:")
    print("=====================")


def unwrap_each_result(result_dict):
    curr_result = result_dict
    curr_title = curr_result['title']
    curr_url = curr_result['formattedUrl']
    curr_snippet = curr_result['snippet']
    return curr_title, curr_url, curr_snippet


def print_result(result_count, curr_title, curr_url, curr_snippet):
    print("Result ", result_count)
    print("[")
    print("URL: ", curr_url)
    print("Title: ", curr_title)
    print("Summary: ", curr_snippet)
    print("]")
    print()


def token(title, snippet):
    title_tokenized = word_tokenize(title)
    snippet_tokenized = word_tokenize(snippet)
    return title_tokenized, snippet_tokenized


def process_doc(title, snippet):
    # make words lower case
    title = title.lower()
    snippet = snippet.lower()
    return title, snippet


def create_vector_matrix(corpus):
    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform(corpus)
    np.set_printoptions(threshold=np.inf)

    ''' print statements to check work '''
    # print(corpus_of_titles_and_snippets_by_document)
    # print(vectorizer.get_feature_names_out())  # print tokens across all documents
    # print(vectorizer.vocabulary_)  # index of each unique word
    # print(bag.toarray())
    ''''''''''''''''''''''''''''''''''''

    return vectorizer.vocabulary_, bag.toarray()


def create_corpus(search_results_dict):
    corpus_of_titles_and_snippets_by_document = []
    corpus_of_titles_by_document = []
    corpus_of_snippets_by_document = []
    for key in search_results_dict.keys():
        concated_processed_documents_snippet_and_title = search_results_dict[key]["processed_title"] + \
                                                         search_results_dict[key]["processed_snippet"]
        corpus_of_titles_and_snippets_by_document.append(concated_processed_documents_snippet_and_title)
        corpus_of_titles_by_document.append(search_results_dict[key]["processed_title"])
        corpus_of_snippets_by_document.append(search_results_dict[key]["processed_snippet"])
    return corpus_of_titles_and_snippets_by_document, corpus_of_titles_by_document, corpus_of_snippets_by_document


# save bag of words by document to search_results_dict; sum up all bags of words into an aggregated bag of words
def bag_of_words_that_aggregates_all_search_results(search_results_dict, token_index, bag_of_words_by_document):
    bag_of_words_all_search_results_summed = []
    for key in search_results_dict.keys():
        # save bag of words by document to search_results_dict
        search_results_dict[key]["bag_of_words_by_document"] = {}
        search_results_dict[key]["bag_of_words_by_document"] = bag_of_words_by_document[key - 1]
        if key == 1:
            bag_of_words_all_search_results_summed = (bag_of_words_by_document[key - 1])
        else:
            bag_of_words_all_search_results_summed = np.add(bag_of_words_by_document[key - 1],
                                                            bag_of_words_all_search_results_summed)
    # combine aggregated bag of words with token index
    token_corpus_debug_checker = {}
    for tkn, tkn_idx in token_index.items():
        token_corpus_debug_checker[tkn] = {'bag_of_words_token_index': tkn_idx,
                                           'corpus_word_count': bag_of_words_all_search_results_summed[tkn_idx]}

    return search_results_dict, bag_of_words_all_search_results_summed, token_corpus_debug_checker

def rocchios(search_results_dict, token_index, query):
    relevant_bow = []
    relevant_docs_ctr=0
    non_relevant_bow = []
    non_relevant_docs_ctr=0
    for key in search_results_dict.keys():                
        if search_results_dict[key]["is_relevant"]:
            #add bow columnwise to relevant_bow
            # print("doc {} is relevant".format(key))
            # print(relevant_bow)
            if relevant_docs_ctr == 0:
                relevant_bow = search_results_dict[key]["bag_of_words_by_document"]
            else:
                relevant_bow = np.add(search_results_dict[key]["bag_of_words_by_document"], relevant_bow)
            relevant_docs_ctr += 1
        else:
            # print("doc {} is not relevant".format(key))
            # print(non_relevant_bow)
            #add bow columnwise to non_relevant_bow
            if non_relevant_docs_ctr == 0:
                non_relevant_bow = search_results_dict[key]["bag_of_words_by_document"]
            else:
                non_relevant_bow = np.add(search_results_dict[key]["bag_of_words_by_document"], non_relevant_bow)
            non_relevant_docs_ctr +=1 

    #print("finding avg")
    relevant_bow = np.divide(relevant_bow, relevant_docs_ctr)
    non_relevant_bow = np.divide(non_relevant_bow, non_relevant_docs_ctr)
    # print(relevant_bow, relevant_docs_ctr)
    # print(non_relevant_bow ,non_relevant_docs_ctr)

    #print("relevant - non_relevant")
    only_relevant_bow = np.subtract(relevant_bow, non_relevant_bow)
    #print(only_relevant_bow)

    #tokenize query
    query = query.lower()
    tokenized_query = query.split()

    #remove query from only_relevant_bow vector
    for query_word in tokenized_query:
        #find col # of query_word
        col_no= token_index[query_word]
        only_relevant_bow[col_no] = -math.inf              

    #find 2 max cols and map to words            
    indices = (-only_relevant_bow).argsort()[:2]
    word1 = list(token_index.keys())[list(token_index.values()).index(indices[0])]
    word2 = list(token_index.keys())[list(token_index.values()).index(indices[1])]
    return word1, word2
        

def main():
    # step 1: receive command line inputs (list of words + target precision value)
    api_key = sys.argv[1]
    engine_key = sys.argv[2]
    precision = float(sys.argv[3])
    query = sys.argv[4]

    # step 2: call api w ip=query, collect  top-10 results to user
    curr_precision = 0.0
    while curr_precision < precision:
        print_search_params(api_key, engine_key, query, precision)

        # get search results
        search_results = get_search_results(query, api_key, engine_key)
        
        if len(search_results) == 0 :
            print("API returned no results")
            sys.exit()

        # unwrap search results json
        json_dump_of_search_results = unwrap_json(search_results)

        # loop through each search result
        search_result_count = 1
        search_results_dict = {}

        # store all document data in a dictionary
        for each_result in json_dump_of_search_results:
            title, url, snippet = unwrap_each_result(each_result)
            processed_title, processed_snippet = process_doc(title, snippet)
            search_results_dict[search_result_count] = {'title': title, "url": url, "snippet": snippet,
                                                        'processed_title': processed_title,
                                                        'processed_snippet': processed_snippet}
            search_result_count += 1

        # create multiple corpus
        corpus_of_titles_and_snippets_by_document, corpus_of_titles_by_document, corpus_of_snippets_by_document = create_corpus(
            search_results_dict)

        # choose which corpus to vectorize and create a bag of words matrix of counts of unique word tokens (columns) by documents (rows)
        token_index, bag_of_words_by_document = create_vector_matrix(
            corpus_of_titles_and_snippets_by_document)  # change the input to vectorize a different corpus
        token_index = {k: v for k, v in
                       sorted(token_index.items(), key=lambda item: item[1])}  # sorts token_index by index

        # save bag of words by document to search_results_dict AND sum up all bags of words into one aggregated bag of words
        search_results_dict, bag_of_words_all_search_results_aggregated, token_corpus_debug_checker = \
            bag_of_words_that_aggregates_all_search_results(search_results_dict, token_index, bag_of_words_by_document)

        '''print statements to check work'''

        # pp.pprint(bag_of_words_by_document)  # "bags of words by document: ",
        # pp.pprint(
        #     bag_of_words_all_search_results_aggregated)  # "bags of words for all search result documents aggregated: ",
        # pp.pprint(
        #     token_corpus_debug_checker)  # "bags of words for all search result documents aggregated with tokens: ",
        # # print(search_results_dict[1]['title'])
        ''''''''''''''''''''''''''''''''''''

        # step 3 loop through dictionary of search results. Print each result to user then get relevance evaluation from user.
        result_count = 1
        yes_counter = 0

        for key in search_results_dict.keys():
            search_results_dict[key]["is_relevant"] = {}
            # step 3, part 1: print
            print_result(result_count, search_results_dict[key]["processed_title"], search_results_dict[key]["url"],
                         search_results_dict[key]["processed_snippet"])

            ##step 3, part 2: Ask user if relevant, increment yes counter and add new key-value pair
            is_relevant = relevant()
            yes_counter = yes_counter + is_relevant
            
            # step 3, part 3: store result in search_results_dict
            search_results_dict[key]["is_relevant"] = is_relevant
            result_count += 1
            
            '''print statements to check work'''
            #pp.pprint(search_results_dict[key])
            ''''''''''''''''''''''''''''''''''''

        curr_precision = yes_counter/result_count
        print("=============================")
        print("FEEDBACK SUMMARY")
        print("Query", query)
        print("Precision", curr_precision)

        if curr_precision == 0:
            print("Precision = 0; terminating")
            sys.exit()

        if (curr_precision > precision):
            print("Desired precision reached, done")
            sys.exit()
        else:
            print("Still below the desired precision of ", precision)
            word1, word2 = rocchios(search_results_dict, token_index, query)

            #TODO: add ordering here
            new_query = query + " " + word1 + " " + word2
            query = new_query
            print("Augmented by ", word1, word2)

       
'''
the following code worked to tokenize but I found a better way so archiving this code 
    #tokenize 
    title_tokenized, snippet_tokenized = token(title, snippet)
    doc_tokenized = title_tokenized + snippet_tokenized
    text = " ".join(doc_tokenized)
    print("tokenize: ", text)
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(text).toarray()
    print(X)
'''
main()

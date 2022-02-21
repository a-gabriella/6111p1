from googleapiclient.discovery import build
import pprint
import sys
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller

from sklearn.feature_extraction.text import CountVectorizer

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
    #call api with query and return result
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query,cx=engine_id,).execute()
    return res

def unwrap_json(search_results):
    top_10_results_json_dump = search_results['items']
    #print("top_10_results", top_10_results_json_dump)
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

def main():
    #step 1: receive command line inputs (list of words + target precision value)
    api_key = sys.argv[1]
    engine_key = sys.argv[2]
    precision = float(sys.argv[3])
    query = sys.argv[4]
    print_search_params(api_key, engine_key, query, precision)

    # step 2: call api w ip=query, collect  top-10 results to user
    curr_precision = 0.0
    while curr_precision < precision:
        #get search results
        search_results = get_search_results(query, api_key, engine_key)

        #unwrap search results json
        json_dump_of_search_results = unwrap_json(search_results)

        # loop through each search result
        search_result_count = 1
        search_results_dict = {}

        # store all document data in a dictionary
        for each_result in json_dump_of_search_results:
            title, url, snippet = unwrap_each_result(each_result)
            processed_title, processed_snippet = process_doc(title, snippet)
            search_results_dict[search_result_count] = {'title': title, "url": url, "snippet": snippet, 'processed_title': processed_title,
                                                        'processed_snippet': processed_snippet}
            search_result_count += 1

        #print statements to check work
        #search_results_dict[count]["bow"] = {}
        #print(search_results_dict)
        #print(search_results_dict[1]['title'])

        # create multiple corpus and then choose one to vectorize
        corpus_of_titles_and_snippets_by_document = []
        corpus_of_titles_by_document = []
        corpus_of_snippets_by_document = []
        for key in search_results_dict.keys():
            concated_processed_documents_snippet_and_title = search_results_dict[key]["processed_title"] + search_results_dict[key]["processed_snippet"]
            corpus_of_titles_and_snippets_by_document.append(concated_processed_documents_snippet_and_title)
            corpus_of_titles_by_document.append(search_results_dict[key]["processed_title"])
            corpus_of_snippets_by_document.append(search_results_dict[key]["processed_snippet"])
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus_of_titles_and_snippets_by_document) #to use just titles or just snippits, change the input to this function
        np.set_printoptions(threshold=np.inf)

        # print statements to check work 
        # print(corpus_of_titles_and_snippets_by_document)
        print(vectorizer.get_feature_names_out())
        print(X.toarray())

        #step 3 loop through dictionary of search results. Print each result then get relevance evaluation from user.
        result_count = 1
        yes_counter = 0
        for key in search_results_dict.keys():
            #step 3, part 1: print
            print_result(result_count, search_results_dict[key]["processed_title"], search_results_dict[key]["url"], search_results_dict[key]["processed_snippet"])

            ##step 3, part 2: Ask user if relevant, increment yes counter and add new key-value pair
            is_relevant = relevant()
            yes_counter = yes_counter + is_relevant
            each_result["is_relevant"] = is_relevant

            # store result
            result_count += 1

'''
the following code worked to tokenize but I found a better way 
    #tokenize 
    title_tokenized, snippet_tokenized = token(title, snippet)
    doc_tokenized = title_tokenized + snippet_tokenized
    text = " ".join(doc_tokenized)
    print("tokenize: ", text)
    matrix = CountVectorizer(max_features=1000)
    X = matrix.fit_transform(text).toarray()
    print(X)
'''

'''
        vectorizer = CountVectorizer()
        bag = vectorizer.fit_transform(dict)
        # Get unique words / tokens found in all the documents. The unique words / tokens represents
        # the features
        print(vectorizer.get_feature_names())
        #
        # Associate the indices with each unique word
        print(vectorizer.vocabulary_)
        #
        # Print the numerical feature vector
        print(bag.toarray())
'''


main()
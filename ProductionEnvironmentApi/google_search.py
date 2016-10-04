

try:
        from googlesearch import GoogleSearch

except ImportError:
        pip install https://pypi.python.org/packages/py2/g/googlesearch/googlesearch-0.7.0-py2-none-any.whl


from pprint import pprint
import wikipedia
import re


def get_google_result(query):
        '''  FUCN return {content:' ', url: ''} for google query search '''
        search_set = set()
        gs = GoogleSearch(query)
        for hit in gs.top_results():
                result = hit['titleNoFormatting']
                #pprint(result)
                if '|' in result:
                        que = hit['titleNoFormatting'].split('|')
                if '-' in result:
                        que = hit['titleNoFormatting'].split('-')
                search_set.add(que[0])
        print search_set
        for search in search_set:
                get_wiki_result(search)

        print '*'*20+'Google End'+'*'*20

def get_wiki_result(query):
        print '*'*20+'WIKI'+'*'*20
        #_wp_list =  wikipedia.search(query)
        _wp_list =  wikipedia.summary(query, sentences=25)
        print _wp_list

        print '*'*20+'WIKI End'+'*'*20

get_google_result('cheken Tkka')
#get_wiki_result('chekeeenuuuuu Tkka')

#!/usr/bin/env python
"""
https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-all-field.html
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/mapping-geo-point-type.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-core-types.html#copy-to
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/mapping-multi-field-type.html
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/mapping-core-types.html
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/analysis-keyword-tokenizer.html
all the filters
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/analysis-elision-tokenfilter.html
https://www.elastic.co/guide/en/elasticsearch/reference/0.90/analysis-shingle-tokenfilter.html

"""

#test Cases

#Eatery raw has keyword analyzer implemented on it, so it will match you exact name
search_for_exact_eatery {                                                 
                            "query":{
                                    "term":{
                                                "eatery_raw":  "hauz khas Social"}},
                            
                            "from": 0,
                            "size": number_of_dishes,
                          }

search_body = {                                                 
                            "query":{
                                    "match":{
                                                "eatery_shingle":  "hauz khas"}},
                            
                            "from": 0,
                            "size": number_of_dishes,
                          }


search_body = {                                                 
                            "query":{
                                    "match":{
                                                "eatery_shingle":  "hauz"}},
                            
                            "from": 0,
                            "size": number_of_dishes,
                          }



search_body = {                                                 
                            "query":{
                                    "match":{
                                                "eatery_shingle":  "khas social"}},
                            
                            "from": 0,
                            "size": number_of_dishes,
                          }


## To search for exact dish
search_body =  {                    
                            "query":{
                                    "term":{
                                                "dish_raw":  "per peri chicken sizzler"}},

                            "from": 0,
                            "size": number_of_dishes,
                          }


##To search for dishes who have something silimar to the queried dish
search_body =  {
                            "query":{
                                    "match":{
                                                "dish_shingle":  "per peri chicken sizzler"}},

                            "from": 0,
                            "size": number_of_dishes,
                          }



##To search with phoneti algorithms
search_body = {                                                 
  "query": {
    "match": {
      "dish_phonetic": {
        "query": "pari pari",
        "fuzziness": 10,
        "prefix_length": 1
      }
    }
  }
}


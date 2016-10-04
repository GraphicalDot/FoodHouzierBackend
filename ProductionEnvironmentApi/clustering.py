#!/usr/bin/env python
"""
This is to test new clustring algortihms
"""
import sys
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import nltk
stemmer = SnowballStemmer("english")

class SimilarityMatrices:

        @staticmethod
        def levenshtein_ratio(__str1, __str2):
                ratio = 'Levenshtein.ratio("{1}", "{0}")'.format(__str1, __str2)
                return eval(ratio)


        @staticmethod
        def __modified_dice_cofficient(coord):
		i, j = coord
		__str1, __str2 = keys[i], keys[j]
                print i, j, __str1, __str2
		__str1, __str2 = __str1.replace(" ", ""), __str2.replace(" ", "")
                __ngrams = lambda __str: ["".join(e) for e in list(nltk.ngrams(__str, 2))]
                __l = len(set.intersection(set(__ngrams(__str1)), set(__ngrams(__str2))))
                total = len(__ngrams(__str1)) + len(__ngrams(__str2))
                """
                if len(set.intersection(set(__str1.split(" ")), set(__str2.split(" ")))) \
                        >= min(len(__str1.split(" ")), len(__str2.split(" "))):
                        print "New matric found  beyween %s and %s\n"%(__str1, __str2)
                        return 0.9
                """
                try:
                    return  float(__l*2)/total
                except Exception as e:
                    return 0






def run(keys):
        stems = [stemmer.stem(t) for t in keys]
        clusters = list()
        X = np.zeros((len(keys), len(keys)), dtype=np.float)
        for i in xrange(0, len(keys)):
                for j in xrange(0, len(keys)):
                        if i == j:
                                #If calculating for same element
                                X[i][j] = 0.5
                                X[j][i] = 0.5

                        if X[i][j] == 0:
                                #st = 'Levenshtein.ratio("{1}", "{0}")'.format(self.keys[i], self.keys[j])
                                #ratio = eval(st)
                                #ratio = SimilarityMatrices.levenshtein_ratio(self.keys[i], self.keys[j])
                                ratio = modified_dice_cofficient(keys[i], keys[j])
                                X[i][j] = ratio
                                X[j][i] = ratio


        #Making tuples of the indexes for the element in X where the rtion is greater than .76
        #indices = np.where((X > .75) & (X < 1))
        indices = np.where(X > .75)
        new_list = zip(indices[0], indices[1])


	a = dict()
	for (i, j) in new_list:
		__list = (list(), a.get(i))[a.get(i) != None]
		__list.append(j)
		print __list
		a[i] =  __list
		print (i, j), a


        found = False
        for e in new_list:
                for j in clusters:
                        if bool(set.intersection(set(e), set(j))):
                                j.extend(e)
                                found = True
                                break
                if not found:
                        clusters.append(list(e))
                        found = False

                found = False 
        original_clusters = clusters

        clusters = [keys[i] for i in original_clusters]
        print clusters
        print clusters

if __name__ == "__main__":
        try:
                print keys
        except Exception as e:
                print e
                
                

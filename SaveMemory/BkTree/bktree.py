#!/usr/bin/env python
"""
This module implements Burkhard-Keller Trees (bk-tree).  bk-trees
allow fast lookup of words that lie within a specified distance of a
query word.  For example, this might be used by a spell checker to
find near matches to a mispelled word.
The implementation is based on the description in this article:
http://blog.notdot.net/2007/4/Damn-Cool-Algorithms-Part-1-BK-Trees
Licensed under the PSF license: http://www.python.org/psf/license/
- Adam Hupp <adam@hupp.org>
"""
from itertools import imap, ifilter
import nltk
import time
from compiler.ast import flatten
import pickle
from sklearn.externals import joblib

class SimilarityMatrices(object):

        @staticmethod
        def levenshtein_ratio(__str1, __str2):
                ratio = 'Levenshtein.ratio("{1}", "{0}")'.format(__str1, __str2)
                return eval(ratio)


        @staticmethod
        def modified_dice_cofficient(__str1, __str2):
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
                    return  100 -int(float(__l*2)/total*100)
                except Exception as e:
                    return 0






class BKTree(object):
        def __init__(self, words):
                """
                Create a new BK-tree from the given distance function and
                words.
        
                Arguments:
                    distfn: a binary function that returns the distance between
                    two words.  Return value is a non-negative integer.  the
                    distance function must be a metric space.
        
                words: an iterable.  produces values that can be passed to
                distfn
        
                """
                self.distfn = self.modified_dice_cofficient

                it = iter(words)
                root = it.next()
                self.tree = (root, {})

                for i in it:
                        self._add_word(self.tree, i)
                self.cluster_tree = self.tree
        
        def modified_dice_cofficient(self, __str1, __str2):
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
                    return  100 -int(float(__l*2)/total*100)
                except Exception as e:
                    return 0





        def _add_word(self, parent, word):
                """
                when first word is being added after the root
                pword = root
                children = {}
                d = distance between rot and word , lets say 76
                tree = (root, {76: ("a": {}) })

                """
                pword, children = parent
                d = self.distfn(word, pword)
                if d in children:
                        self._add_word(children[d], word)
                else:
                        children[d] = (word, {})

        def query(self, word, n):
                """
                Return all words in the tree that are within a distance of `n' from `word`.  
                Arguments:
                        word: a word to query on
                            n: a non-negative integer that specifies the allowed distance 
                            from the query word.  
        
                Return value is a list of tuples (distance, word), sorted in 
                ascending order of distance.
                """
                def rec(parent):
                        pword, children = parent
                        d = self.distfn(word, pword)
                        results = []
                        if d <= n:
                            results.append(pword)
                
                        for i in range(d-n, d+n+1):
                            child = children.get(i)
                            if child is not None:
                                    results.extend(rec(child))
                        return results

                # sort by distance
                return sorted(rec(self.cluster_tree))
    


def brute_query(word, words, distfn, n):
        """A brute force distance query
        Arguments:
            word: the word to query for
            words: a iterable that produces words to test
            distfn: a binary function that returns the distance between a
            `word' and an item in `words'.
            n: an integer that specifies the distance of a matching word
    
        """
        return [i for i in words if distfn(i, word) <= n]


def maxdepth(tree, count=0):
        _, children = t
        if len(children):
            return max(maxdepth(i, c+1) for i in children.values())
        else:
            return c


# http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
def levenshtein(s, t):
        m, n = len(s), len(t)
        d = [range(n+1)]
        d += [[i] for i in range(1,m+1)]
        for i in range(0,m):
                for j in range(0,n):
                        cost = 1
                        if s[i] == t[j]: cost = 0

                        d[i+1].append( min(d[i][j+1]+1, # deletion
                               d[i+1][j]+1, #insertion
                               d[i][j]+cost) #substitution
                                   )
        return d[m][n]


def dict_words(dictfile="/usr/share/dict/american-english"):
    return ifilter(len,
                   imap(str.strip,
                        open(dictfile)))


def timeof(fn, *args):
        import time
        t = time.time()
        res = fn(*args)
        print "time: ", (time.time() - t)
        return res



if __name__ == "__main__":

        keys = ["book", "cake", "boo", "bool", "cool", "books", "cart", "cape", "cook", "boo"]
        tree_instance = BKTree(keys)
        distance = 20
        """
        tree_instance = BKTree(levenshtein , keys)
        tree = tree_instance.cluster_tree
        """
        with open("pickled_tree.pickle", 'wb') as handle:
              pickle.dump(tree_instance.cluster_tree, handle)

        clusters = list()
        for dish in keys:
                if flatten(clusters) == keys:
                    break
                result = tree_instance.query(dish, distance)
                print result
                clusters.append(result)
                if result:
                        keys =  list(set.symmetric_difference(set(keys), set(result)))
                        """
                        keys.remove(dish)
                        for e in result:
                                try:
                                        keys.remove(e)
                                        print "removed %s"%e
                                except Exception as error:
                                        print error
                                        continue
                        """
                else:
                        keys.remove(dish)
                        result.append([dish])
                print "clusters %s"%clusters
                print "keys %s"%keys
        print clusters, "\n\n"
        print keys
        print tree_instance.query("book", distance)

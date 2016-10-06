#!/usr/bin/env python
##http://limeandpepper.tumblr.com/post/1372362277/bk-trees-in-python
"""
The whole tree is actually a single tuple of (word, children). Children is a dictionary of tuples with keys being the distance of each child to its parent. 
So, we end up with one big ass nested tuple structure. For example, if I was to create a tree having as root the word “cat” and then add “bat”, “cats”, “bats”, 
I would end up with the following structure:

    ("cat", 
          { 1: ("bat", 
                     { 2: ("cats", {}) }),
            2: ("bats", {})
    )


"""

class BkTree:
     "A Bk-Tree implementation."
     
     def __init__(self, root, distance_function):
         self.df = distance_function
         self.root = root 
         self.tree = (root, {})
     
     def build(self, words):
         "Build the tree."
         for word in words:
             self.tree = self.insert(self.tree, word)

     def insert(self, node, word):
         "Inserts a word in the tree."
         d = self.df(word, node[0])
         if d not in node[1]:
             node[1][d] = (word, {})
         else:
             self.insert(node[1][d], word)
         return node

     def query(self, word, n):
         "Returns a list of words that have the specified edit distance from the search word."
         def search(node):
             d = self.df(word, node[0])
             results = []
             if d == n:
                 results.append(node[0])
             for i in range(d-n, d+n+1):
                 children = node[1]
                 if i in children:
                     results.extend(search(node[1][i]))
             return results
         
        root = self.tree
        return search(root)



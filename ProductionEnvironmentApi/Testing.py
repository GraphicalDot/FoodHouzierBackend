#!/usr/bin/env python

import os
import sys

import nltk
from topia.termextract import extract   
from textblob.np_extractors import ConllExtractor 
from topia.termextract import extract 
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))

from Text_Processing.Sentence_Tokenization import CopiedSentenceTokenizer, SentenceTokenizationOnRegexOnInterjections


class Test(object):
        

        def __init__(self, text):
                self.text = text
                self.conll_extractor = ConllExtractor()
                self.topia_extractor = extract.TermExtractor()
                
                ##Our custom tokenizer
                self.custom_sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                self.tokenized_sentences = self.custom_sent_tokenizer.tokenize(self.text)
                
                ##This method will apply the sstemmers to the sentences
                self.stemming()

                print nltk.sent_tokenize(self.text)
                self.np_textblob()
                self.np_topia()


        def np_topia(self):
                print "\n\n From text topia extractor"
                for sentence in self.tokenized_sentences:
                        noun_phrases = self.topia_extractor(sentence)
                        print sentence, "\t", noun_phrases, "\n"

                

        def stemming(self):
                """
                Applying snowball stemmer to the the words so that the classification works in a better way
                """
                self.tokenized_sentences = [" ".join([SnowballStemmer("english").stem(word) for word in nltk.wordpunct_tokenize(sent)]) for sent in self.tokenized_sentences]
                return 


        def lemmatize(self):
                """


                """
                lmtzr = WordNetLemmatizer()
                self.lemmatize_sentences = [" ".join([lmtzr.lemmatize(word) for word in nltk.wordpunct_tokenize(sent)]) for sent in self.tokenized_sentences]
                return 



        def np_textblob(self):
                print "From text blob np extractor"
                for sentence in self.tokenized_sentences:
                        blob = TextBlob(sentence, np_extractor=self.conll_extractor)
                        print sentence, "\t", blob.noun_phrases, "\n"

        def ner(self):
                nltk.ne_chunk()


if __name__ == "__main__":
        Test("""A landmark and a legendary restaurant smack in the middle of Delhi. An icon and a superstar of restaurants. Many Open a restaurant imagining one day they will own a joint as successful as Gulati. I have been coming to eat here for as long a I can remember. I went here last month with my friend and we ordered 2 biryanis and a non veg platter. The biryani was nice and fragrant with tender chicken pieces and the mutton biryani was sensational. The non veg platter was finished in 5 mins and we were wishing to have more but wanted to keep room in our stomach for the Biryani. I knew my day was made as and when we decided we were going to Gulati for dinner.""")































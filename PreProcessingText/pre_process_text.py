
#!/usr/bin/env python

import re
from blessings import Terminal
t = Terminal()


class PreProcessText(object):
        def __init__(self):
                pass


        @staticmethod
        def process(text):
                text = PreProcessText.convert_lowercase(text)
                text = PreProcessText.remove_all_instances(text)
                text = PreProcessText.remove_and_replace(text)
                return text


        
        @staticmethod
        def remove_numbers(text):
                return re.sub('\d+', '', text)
        
        @staticmethod
        def convert_lowercase(text):
                return text.lower()
        
        

        @staticmethod
        def remove_and_replace(text):
                """
                Remove mupltiple occurences and replace it with single
                occurence 
                """
                text = re.sub('\)+', ')', text)
                text = re.sub("\.{2,10000}", " ", text) 
                text = re.sub("\?{2,10000}", " ", text) 
                text = re.sub("\!{2,10000}", " ", text) 
                return text



        @staticmethod
        def remove_all_instances(text):
                """
                remove every occurenc of #, - and any number from the text, so
                for example -----, -, #, ##, or number will be replaced by
                a whitespace in the text in consideration
                """
                
                text = re.sub("-+", " ", text)
                text = re.sub("_+", " ", text)
                text = re.sub("&amp;", " ", text)
                text = re.sub("&amp", " ", text)
                text = re.sub("&nbsp;", " ", text)
                text = re.sub("&nbsp", " ", text)
                text = re.sub("\*+", "", text)
                text = re.sub("#+", " ", text)
                #text = re.sub("\d+.\d+/\d+", " ", text)
                text = re.sub("[4,5]\s{0,2}/\s{0,2}5", " good", text)
                text = re.sub("[7,10]\s{0,2}/\s{0,2}10", " good", text)
                
                text = re.sub("3\s{0,2}/\s{0,2}5", " average", text)
                text = re.sub("[4,6]\s{0,2}/\s{0,2}10", " average", text)
                
                text = re.sub("[1,2]\s{0,2}/\s{0,2}5", " good", text)
                text = re.sub("[1,3]\s{0,2}/\s{0,2}10", " good", text)

                re.sub("\d+/\d+", " ", text)
                text = re.sub("\d+", " ", text)

                
                #text = re.sub("\d+.\d+/\d+", " ", text)
                
                
                text = re.sub("\d+/\d+", " ", text)
                return text

##Todo: New way to have space after fullstop


def test():
        """
        Test the output of the class PreProcessText
        """
        import pymongo
        import random
        connection = pymongo.MongoClient()
        reviews = connection.Reviews.ZomatoReviews
        j = random.choice(range(20))
        i = 0
        for post in reviews.find().skip(j):     
                text = post.get("review_text")
                _text = PreProcessText.convert_lowercase(text)
                _text = PreProcessText.remove_all_instances(_text)
                _text = PreProcessText.remove_and_replace(_text)
                print text, "\n", t.red(_text), "\n\n"
                j += random.choice(range(20))
                i += 1
                if i == 100:
                    break
                        

        return 





if __name__ == "__main__":
        #test()
        import pymongo
        from blessings import Terminal
        terminal = Terminal()
        connection = pymongo.MongoClient()
        training_data = connection.training_data
        sentiment_collection = training_data.training_sentiment_collection
		
        _list = list()
        for post in sentiment_collection.find():
            _list.append((post.get("sentiment"), post.get("sentence")))
	
                         
        sentiments = list()
        sentiment_list = list(set(_list))
        for (sentiment, sentence) in sentiment_list:
                text = PreProcessText.remove_all_instances(sentence)
                text = PreProcessText.remove_and_replace(text)
                print sentiment
                print terminal.red(sentence)
                print terminal.green(text), "\n\n"
                sentiments.append((sentiment, text))










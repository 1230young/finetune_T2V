from summa import keywords
from keybert import KeyBERT
def get_keywords_TextRank(text,num=10):
    return keywords.keywords(text,scores=True)


def get_keywords_keybert(text,num=10):
    kw_model = KeyBERT(model='models/distilbert-base-nli-mean-tokens')
    
    keywords = kw_model.extract_keywords(text, 
                                        keyphrase_ngram_range=(1, 2), 
                                        stop_words='english', 
                                        highlight=False, 
                                        top_n=num) 
    
    keywords_list= list(dict(keywords).keys())
    return keywords_list



if __name__ == "__main__":
    text = "a man and woman standing next to each other in a crowd of people"

    print(get_keywords_TextRank(text))
    print("ok")
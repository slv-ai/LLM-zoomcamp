import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Index:
    #text_fields are fields containing the main textual content of the documents.
    #keyword_fields are fields containing keywords for filtering, such as course names.

    def __init__(self,text_fields,keyword_fields,vectorizer_params={}):
        self.text_fields=text_fields
        self.keyword_fields=keyword_fields
        self.vectorizers={field:TfidfVectorizer(**vectorizer_params)for field in text_fields}
        self.keyword_df=None
        self.text_matrices={}
        self.docs=[]

    def fit(self,docs):
        self.docs=docs
        keyword_data={field:[] for field in self.keyword_fields}
        for field in self.text_fields:
            texts=[doc.get(field,'')for doc in docs]
            self.text_matrices[field]=self.vectorizers[field].fit_transform(texts)
        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field,''))
        self.keyword_df=pd.DataFrame(keyword_data)
        return self
    def search(self,query,filter_dict={},boost_dict={},num_results=10):
        query_vect={field:self.vectorizers[field].transform([query])for field in self.text_fields}
        scores=np.zeros(len(self.docs))
        #compute cosine similarity for each text field and apply boost
        for field,query_v in query_vect.items():
            similar=cosine_similarity(query_v,self.text_matrices[field]).flatten()
            boost=boost_dict.get(field,1)
            scores += similar * boost
        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs

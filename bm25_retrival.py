import jieba
from gensim.summarization import bm25

class BM25Retrieval:
    def __init__(self, path):
        self.path = path
        self.docs = self.load_corpus(self.path)
        self.bm25Model = bm25.BM25([jieba.lcut(line) for line in self.docs])


    def load_corpus(self, path):
        with open(path, encoding='utf-8') as fp:
            words_list = []
            for line in fp:
                words_list.append(line)

            return words_list

    def retrieval(self, query, top_k):
        scores = self.bm25Model.get_scores(jieba.lcut(query))
        match_score = {e: s for e, s in zip(self.docs, scores)}
        match_score = sorted(match_score.items(), key=lambda x : x[1], reverse=True)
        return [i[0] for i in match_score[:top_k]]

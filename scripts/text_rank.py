import torch
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from flask import Flask, request

app = Flask(__name__)

model_dir = '../bert/ms'
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert_model = BertModel.from_pretrained(model_dir)


def text_rank(text, max_seq_length=512):
    tokens = tokenizer.tokenize(text)
    max_seq_length -= 2
    if len(tokens) <= max_seq_length:
        return text
    sentences = list(map(lambda item: item + '。', filter(lambda item: item, text.split('。'))))
    sen_tokens = []
    for sentence in sentences:
        sen_tokens.append(
            bert_model.embeddings.word_embeddings(
                torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)).clone().detach().mean(dim=1).numpy())
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim_mat[j][i] = cosine_similarity(sen_tokens[i], sen_tokens[j])
    nx_graph = nx.from_numpy_array(sim_mat)
    try:
        scores = nx.pagerank(nx_graph)
    except nx.exception.PowerIterationFailedConvergence:
        return tokens[len(tokens) - max_seq_length:]
    ranked_sentences = sorted([(scores[i], i, s) for i, s in enumerate(sentences)], reverse=True)
    collected = []
    length = 0
    for sentence in ranked_sentences:
        if length + len(sentence[2]) <= max_seq_length:
            collected.append(sentence)
            length += len(sentence[2])
    ranked_sentences = sorted(collected, key=lambda x: x[1])
    return ''.join([i[2] for i in ranked_sentences])


@app.route('/text_rank', methods=['POST'])
def process_text():
    text = request.get_data(as_text=True)
    return text_rank(text)


if __name__ == '__main__':
    app.run()

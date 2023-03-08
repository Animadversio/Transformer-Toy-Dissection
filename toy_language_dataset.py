
#%% An artificial language made from a probabilistic generative grammar
#%% list of words and their parts of speech
# nouns
noun_list = ["cat", "dog", "fox", "bird", "horse", "sheep", "cow", "bear", "zebra", "giraffe"]
# intransitive verb
intrans_verb_list = ["ran", "jumped", "swam", "flew", "walked", "slept", "sat", "stood", "danced"]
# transitive verbs that took an object or a clause
trans_verbs_list = ["saw", "heard", "smelled", ]
# adjectives
adj_list = ["big", "small", "tall", "short", "long", "wide", "fat", "thin", "round", "square", "smart", "pretty"]
# adverbs
article_list = ["the", "a"]
# conjunctive that introduces a clause.
conjunction_list = ["that"]
#%%
# Rules for mapping part of speech to words
word_map = {
    "N": noun_list,
    "IV": intrans_verb_list,
    "TV": trans_verbs_list,
    "Adj": adj_list,
    "Article": article_list,
    "Conj": conjunction_list,
}
# Grammar for generating sentences
rules = {
    # sentence
    "S": [["NP", "VP"]],
    # noun phrase
    "NP": [["Article", "N"], ["Article", "Adj", "N"], ["Article", "Adj", "Adj", "N"], ["Article", "N", "Conj", "IV"]],
    # verb phrase
    "VP": [["IV"], ["TV", "NP"], ["TV", "Conj", "NP", "VP"], ],
}
#%%
import random
def generate_sentences(rules, word_map, max_depth=3, show=False):
    """ A sentence generator with probabilistic generative grammar. """
    initial_token = "S"
    sentence = [initial_token]
    depth = 0
    while True:
        next_sentence = []
        fully_expanded = True
        for token in sentence:
            if token in rules:
                # expand the phrase
                if depth < max_depth:
                    next_sentence.extend(random.choice(rules[token]))
                else:
                    # to limit complexity, we stop adding clauses
                    next_sentence.extend(random.choice(rules[token][:-1]))  # don't expand into the last conjunctive rule

                fully_expanded = False
            else:
                next_sentence.append(token)

        sentence = next_sentence
        depth += 1
        if show:
            print(sentence)
        if fully_expanded:
            break
    # turn tokens into words
    verbal_sentence = []
    for token in sentence:
        word = random.choice(word_map[token])
        verbal_sentence.append(word)
    sent_str = " ".join(verbal_sentence)
    return verbal_sentence, sent_str


word_seq, sentence_str = generate_sentences(rules, word_map, show=False, max_depth=3)
print(sentence_str)

#%%
full_word_set = set(sum([words for words in word_map.values()], []))
dictionary = {word: i for i, word in enumerate(full_word_set)}  # Mapping words to indices
dictionary["[End]"] = len(dictionary)
EOS_ID = dictionary["[End]"]  # end of sentence token
PAD_ID = len(dictionary)  # padding token
dictionary[""] = PAD_ID
inverse_dictionary = {i: word for word, i in dictionary.items()}   # Mapping indices to words

#%%
def tokenize_sentence(sentence):
    """ Tokenize a sentence into a list of words. """
    word_seq = sentence.split(" ")
    return word_seq


def encode_sentence(sentence):
    """ Encode a sentence into a list of indices. """
    word_seq = tokenize_sentence(sentence)
    inds = encode2indices(word_seq)
    return inds


def encode2indices(word_seq):
    """ Encode a list of words into a list of indices. """
    inds = [dictionary[word] for word in word_seq]
    return inds


def decode2words(indices):
    """ Decode a list of indices into a list of words. """
    words = [inverse_dictionary[ind] for ind in indices]
    return words


def decode2sentence(indices):
    """ Decode a list of indices into a sentence. """
    words = decode2words(indices)
    sentence = " ".join(words)
    return sentence

#%%
import torch
from torch.nn.utils.rnn import pad_sequence
def batch_sampler(batch_size=32, max_len=128):
    batch = []
    for i in range(batch_size):
        word_seq, _ = generate_sentences(rules, word_map)
        word_seq.append("[End]")
        inds = encode2indices(word_seq)
        batch.append(torch.tensor(inds, dtype=torch.long))
    # pad the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    # chuck to the max_len
    padded_batch = padded_batch[:, :max_len]
    return padded_batch

#%%
for i in range(5):
    batch = batch_sampler()
    print(batch.shape)
    for j in range(batch.shape[0]):
        print(decode2sentence(batch[j].tolist()))
#%%
import torch
from torch.optim import AdamW
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
#%%
miniGPTconfig = GPT2Config(vocab_size=len(dictionary), n_positions=128, n_ctx=128, n_embd=64, n_layer=4, n_head=8,
                           eos_token_id=EOS_ID, pad_token_id=PAD_ID)
miniGPT = GPT2LMHeadModel(miniGPTconfig, )
#%%
optimizer = AdamW(miniGPT.parameters(), lr=5e-4)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
miniGPT.train()
miniGPT.to(device)
batch_size = 512
for epoch in range(5):
    for i in range(100):
        batch = batch_sampler(batch_size=batch_size)
        output = miniGPT(batch.to(device), labels=batch.to(device), )
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch}, batch {i}, loss {loss.item()}")


miniGPT.eval().to("cpu")

#%%
prompt = "the dog"
prompt_inds = encode_sentence(prompt)
ind_tsr = miniGPT.generate(torch.tensor(prompt_inds).long()[None, :], max_length=32, do_sample=True, top_k=0, top_p=0.9, temperature=0.7,
                           num_return_sequences=1, pad_token_id=PAD_ID)
print(decode2sentence(ind_tsr[0].tolist()))


#%% Your own sampling function
def GPT_generate_sentence(model, max_len=32):
    """ Generate a sentence using the model. """
    model.eval()
    with torch.no_grad():
        # start with a random token
        token = torch.randint(0, len(dictionary), (1, 1))
        sentence = [token]
        for i in range(max_len):
            output = model(token)
            logits = output[0]
            next_token = torch.argmax(logits, dim=-1)
            sentence.append(next_token)
            token = next_token
    sentence = torch.cat(sentence, dim=0)
    sentence = sentence.squeeze().tolist()
    sentence = decode2sentence(sentence)
    return sentence
#%%
token_embedding = miniGPT.transformer.wte.weight
position_embedding = miniGPT.transformer.wpe.weight
#%% visulize the token embedding clusters
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

tsne = TSNE(n_components=2, random_state=0)
token_embedding_2d = tsne.fit_transform(token_embedding.detach().numpy())
kmeans = KMeans(n_clusters=6, random_state=0).fit(token_embedding_2d)
plt.scatter(token_embedding_2d[:, 0], token_embedding_2d[:, 1], c=kmeans.labels_)
# annotate each word on the plot
for i, word in enumerate(inverse_dictionary.values()):
    plt.annotate(word, (token_embedding_2d[i, 0], token_embedding_2d[i, 1]))
plt.show()
#%%
nCluster = 7
kmeans2 = KMeans(n_clusters=nCluster, random_state=42).fit(token_embedding.detach().numpy())
# print the list of words in each cluster
for icluster in range(nCluster):
    cluster_words = [word for word, ind in dictionary.items() if kmeans2.labels_[ind] == icluster]
    print(f"cluster {icluster}: {cluster_words}")

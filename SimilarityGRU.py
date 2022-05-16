import torch
from datetime import datetime
import argparse
import random
import pickle
import codecs
import json
import os
import nltk
import numpy as np
from pprint import pprint
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable
import csv
import pandas as pd

# ==================== Datasets ==================== #

w2i = {}
i2w = []
PADDING_SYMBOL = ' '
w2i[PADDING_SYMBOL] = 0
i2w.append(PADDING_SYMBOL)
UNK_SYMBOL = '<UNK>'
w2i[UNK_SYMBOL] = 1
i2w.append(UNK_SYMBOL)


def load_glove_embeddings(embedding_file):
    N = len(w2i)
    embeddings = [0] * N
    with codecs.open(embedding_file, 'r', 'utf-8') as f:
        for line in f:
            data = line.split()
            word = data[0].lower()
            if word not in w2i:
                w2i[word] = N
                i2w.append(word)
                N += 1
                embeddings.append(0)
            vec = [float(x) for x in data[1:]]
            D = len(vec)
            embeddings[w2i[word]] = vec
    # Add a '0' embedding for the padding symbol
    embeddings[0] = [0] * D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for word in w2i:
        index = w2i[word]
        if embeddings[index] == 0:
            embeddings[index] = (np.random.random(D) - 0.5).tolist()
    return D, embeddings


class QuestionsDataset(Dataset):
    def __init__(self, filename, record_symbols=True):
        try:
            nltk.word_tokenize("hi there.")
        except LookupError:
            nltk.download('punkt')
        self.q1_list = []
        self.q2_list = []
        self.target_list = []
        # Read the datafile
        with codecs.open(filename, 'r', 'utf-8') as f:
            lines = csv.reader(f)
            is_first_line = True
            for line in lines:
                if is_first_line:
                    is_first_line = False
                    continue
                # Process question 1
                q1_source_sentence = []
                for w in nltk.word_tokenize(line[1].lower()):
                    if w not in w2i and record_symbols:
                        w2i[w] = len(i2w)
                        i2w.append(w)
                    q1_source_sentence.append(w2i.get(w, w2i[UNK_SYMBOL]))
                self.q1_list.append(q1_source_sentence)
                # Process question 2
                q2_source_sentence = []
                for w in nltk.word_tokenize(line[2].lower()):
                    if w not in w2i and record_symbols:
                        w2i[w] = len(i2w)
                        i2w.append(w)
                    q2_source_sentence.append(w2i.get(w, w2i[UNK_SYMBOL]))
                self.q2_list.append(q2_source_sentence)
                # Process label
                self.target_list.append(int(line[3]))

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        return self.q1_list[idx], self.q2_list[idx], self.target_list[idx]


class PadSequence:

    def __call__(self, batch, padding=w2i[PADDING_SYMBOL]):
        q1, q2, label = zip(*batch)
        max_q1_len = max(map(len, q1))
        max_q2_len = max(map(len, q2))
        max_q_len = max(max_q1_len, max_q2_len)
        padded_q1 = [[q[i] if i < len(q) else padding for i in range(max_q_len)] for q in q1]
        padded_q2 = [[q[i] if i < len(q) else padding for i in range(max_q_len)] for q in q2]
        return padded_q1, padded_q2, label


# ==================== Encoder ==================== #

class Encoder(nn.Module):

    def __init__(self, no_of_input_symbols, embeddings=None, embedding_size=50, hidden_size=50, device='cpu',
                 tune_embeddings=False, threshold_factor=20, stacked_layer=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.stacked_layer = stacked_layer
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(no_of_input_symbols, embedding_size)
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float),
                                                 requires_grad=tune_embeddings)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True, num_layers=stacked_layer)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.activation = nn.Sigmoid()
        self.device = device
        self.threshold_factor = threshold_factor
        self.to(device)

    def set_embeddings(self, embeddings):
        self.embedding.weight = torch.tensor(embeddings, dtype=torch.float)

    def forward(self, x):
        word_embedding_x = self.embedding(torch.tensor(x).to(self.device))
        _, hidden_stacked = self.rnn(word_embedding_x)
        forward_hidden_stacked, backward_hidden_stacked = torch.chunk(hidden_stacked, 2, dim=0)
        forward_hidden = torch.permute(forward_hidden_stacked, (1, 0, 2)).reshape((-1, self.stacked_layer * self.hidden_size))
        backward_hidden = torch.permute(backward_hidden_stacked, (1, 0, 2)).reshape((-1, self.stacked_layer * self.hidden_size))
        q1_hidden, q2_hidden = torch.chunk(torch.cat([forward_hidden, backward_hidden], dim=1), 2, dim=0)
        return self.activation(self.threshold_factor * (self.cos_sim(q1_hidden, q2_hidden) - self.threshold))


def evaluate(ds, encoder):
    confusion_matrix = [[0, 0], [0, 0]]
    padding = w2i[PADDING_SYMBOL]
    for q1, q2, label in ds:
        max_len = max([len(q1), len(q2)])
        padded_q1 = [[q1[i] if i < len(q1) else padding for i in range(max_len)]]
        padded_q2 = [[q2[i] if i < len(q2) else padding for i in range(max_len)]]
        questions = padded_q1 + padded_q2
        score = encoder.forward(questions)[0]
        predict = 1 if score > 0.5 else 0
        confusion_matrix[label][predict] += 1
    table = [['', 'Predicted different', 'Predicted same'],
             ['Real different', confusion_matrix[0][0], confusion_matrix[0][1]],
             ['Real same', confusion_matrix[1][0], confusion_matrix[1][1]]]
    t = AsciiTable(table)
    print(t.table)
    print("Accuracy: {}".format(round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4)))
    precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    f1_score = 2 * precision * recall / (precision + recall)
    print("Precision: {}".format(round(precision, 4)))
    print("Recall: {}".format(round(recall, 4)))
    print("F1 score: {}".format(round(f1_score, 4)))


def process_and_produce(ds, encoder):
    padding = w2i[PADDING_SYMBOL]
    results = []
    for q1, q2, label in ds:
        max_len = max([len(q1), len(q2)])
        padded_q1 = [[q1[i] if i < len(q1) else padding for i in range(max_len)]]
        padded_q2 = [[q2[i] if i < len(q2) else padding for i in range(max_len)]]
        questions = padded_q1 + padded_q2
        score = float(encoder.forward(questions)[0])
        results += [[score, label]]
    return results


if __name__ == '__main__':

    # ==================== Main program ==================== #
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-tr', '--train', default='train.txt', help='A training file')
    parser.add_argument('-te', '--test', default='test.txt', help='A test file')
    parser.add_argument('-ef', '--embeddings', default='', help='A file with word embeddings')
    parser.add_argument('-et', '--tune-embeddings', action='store_true', help='Fine-tune GloVe embeddings')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-hs', '--hidden_size', type=int, default=50, help='Size of hidden state')
    parser.add_argument('-st', '--stacked_layer', type=int, default=1, help='Number of stacked layers')
    parser.add_argument('-bs', '--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('-thf', '--threshold_factor', type=int, default=20, help='The threshold factor for the similarity score')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-s', '--save', action='store_true', help='Save model')
    parser.add_argument('-l', '--load', type=str, help="The directory with encoder and decoder models to load")
    parser.add_argument('-p', '--produce', type=str, help="Process train and test datasets to produce aggregate file (for ensemble learning)")

    args = parser.parse_args()

    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if is_cuda_available else 'cpu'

    if args.load:
        w2i = pickle.load(open(os.path.join(args.load, "w2i"), 'rb'))
        i2w = pickle.load(open(os.path.join(args.load, "i2w"), 'rb'))
        settings = json.load(open(os.path.join(args.load, "settings.json")))

        encoder = Encoder(
            len(i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'],
            tune_embeddings=settings['tune_embeddings'],
            threshold_factor=settings['threshold_factor'],
            stacked_layer = settings['stacked_layer'],
            device=device
        )
        encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder.model")))

        print("Loaded model with the following settings")
        print("-" * 40)
        pprint(settings)
        print()
    else:
        # ==================== Training ==================== #
        # Reproducibility
        random.seed(5719)
        np.random.seed(5719)
        torch.manual_seed(5719)
        torch.use_deterministic_algorithms(True)
        if is_cuda_available:
            torch.backends.cudnn.benchmark = False
        # Read datasets
        training_dataset = QuestionsDataset(args.train)
        print("Number of unique words in training set: ", len(i2w))
        print("Number of training pairs: ", len(training_dataset))
        print()
        # If we have pre-computed word embeddings, then make sure these are used
        if args.embeddings:
            embedding_size, embeddings = load_glove_embeddings(args.embeddings)
        else:
            embedding_size = args.hidden_size
            embeddings = None
        training_loader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=PadSequence())
        criterion = nn.CrossEntropyLoss()
        encoder = Encoder(
            len(i2w),
            embeddings=embeddings,
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
            tune_embeddings=args.tune_embeddings,
            threshold_factor=args.threshold_factor,
            stacked_layer=args.stacked_layer,
            device=device
        )
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        encoder.train()

        print(datetime.now().strftime("%H:%M:%S"), "Starting training...")
        for epoch in range(args.epochs):
            total_loss = 0
            for q1, q2, label in training_loader:
                encoder_optimizer.zero_grad()
                questions = q1 + q2
                predict_scores = encoder(questions)
                loss = torch.sum(torch.abs(predict_scores - torch.tensor(label).to(device))) / args.batch_size
                loss.backward()
                encoder_optimizer.step()
                total_loss += loss
            print(encoder.threshold)
            print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", total_loss.detach().item())

        # ==================== Save the model  ==================== #

        if args.save:
            dt = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
            newdir = 'model_' + dt
            os.mkdir(newdir)
            torch.save(encoder.state_dict(), os.path.join(newdir, 'encoder.model'))
            with open(os.path.join(newdir, 'w2i'), 'wb') as f:
                pickle.dump(w2i, f)
            with open(os.path.join(newdir, 'i2w'), 'wb') as f:
                pickle.dump(i2w, f)
            settings = {
                'training_set': args.train,
                'test_set': args.test,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'hidden_size': args.hidden_size,
                'embedding_size': embedding_size,
                'tune_embeddings': args.tune_embeddings,
                'threshold_factor': args.threshold_factor,
                'stacked_layer': args.stacked_layer
            }
            with open(os.path.join(newdir, 'settings.json'), 'w') as f:
                json.dump(settings, f)

    # ==================== Evaluation ==================== #

    encoder.eval()
    print("Evaluating on the test data...")
    test_dataset = QuestionsDataset(args.test, record_symbols=False)
    print("Number of test pairs: ", len(test_dataset))
    print()
    evaluate(test_dataset, encoder)
    print()

    # ==================== Process files for further ensemble learning ==================== #

    if args.produce:
        print("Processing the training set...")
        if args.load:
            training_dataset = QuestionsDataset(args.train, record_symbols=False)
        pd.DataFrame(process_and_produce(training_dataset, encoder)).to_csv(args.produce + '/SimGRU_train.csv', header=['SimGRU', 'label'], index=False)
        print("Processing the test set...")
        pd.DataFrame(process_and_produce(test_dataset, encoder)).to_csv(args.produce + '/SimGRU_test.csv', header=['SimGRU', 'label'], index=False)

    # ==================== User interaction ==================== #

    while True:
        print("Question 1: ")
        text = input("> ")
        if text == 'q':
            break
        if text == "":
            continue
        try:
            q1 = [w2i[w] for w in nltk.word_tokenize(text.lower())]
        except KeyError:
            print("Erroneous input string")
            continue
        print("Question 2: ")
        text = input("> ")
        if text == 'q':
            break
        if text == "":
            continue
        try:
            q2 = [w2i[w] for w in nltk.word_tokenize(text.lower())]
        except KeyError:
            print("Erroneous input string")
            continue
        max_len = max([len(q1), len(q2)])
        padding = w2i[PADDING_SYMBOL]
        padded_q1 = [[q1[i] if i < len(q1) else padding for i in range(max_len)]]
        padded_q2 = [[q2[i] if i < len(q2) else padding for i in range(max_len)]]
        questions = padded_q1 + padded_q2
        score = float(encoder.forward(questions)[0])
        if score > 0.5:
            print("Same: {}".format(round(score, 4)))
        else:
            print("Different: {}".format(round(score, 4)))

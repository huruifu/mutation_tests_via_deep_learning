import json
import sklearn_pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from keywords import *
from neural_network import *
from transformer import * 


def create_mutation(request, transformer, word_map, max_len=30):
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in request.split(' ')]
    question = torch.LongTensor(enc_qus).unsqueeze(0)
    question_mask = (question!=0).unsqueeze(1).unsqueeze(1)  
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]])
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]])], dim = 1)
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()
    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])
    
    return sentence

def generate_mutation_score(input, nn_model):
    return nn_model.forward(input)

def generate_valid_nn_input(df):
    mapper = sklearn_pandas.DataFrameMapper(
        [
            (
                ["mutationOperator", "parentStmtContextDetailed"],
                OneHotEncoder(handle_unknown="ignore"),
            )
        ])
    return mapper.fit_transform(df.copy()).astype(np.float32)

if __name__ == "__main__":
    with open('WORDMAP_corpus.json', 'r') as j:
        word_map = json.load(j)
    transformer = torch.load('transformer.tar', map_location=torch.device('cpu'))
    nn_model = torch.load('nn.tar', map_location=torch.device('cpu'))
    methodName = "checkA"
    code = "if (a == 1 || b == 2)"
    nestingIf = 0
    nestingLoop = 1
    parentStmtContextDetailed = "VARIABLE"
    
    containDoubleEqual = "==" in code
    for key in keywords_map:
        if key not in code:
            continue
        if key == "=" and containDoubleEqual:
            continue
        # Find keyword
        general_code_expression = keywords_map[key]
        request = f"{methodName} {general_code_expression} {nestingIf} {nestingLoop} {parentStmtContextDetailed}"
        mutation = create_mutation(request, transformer, word_map)
        mutation_operator = f"{general_code_expression}:{mutation}"
        data = {'mutationOperator': [mutation_operator], 'parentStmtContextDetailed': [parentStmtContextDetailed]}  
        df = pd.DataFrame(data)
        input = generate_valid_nn_input(df)
        mutation_score = generate_mutation_score(input)
        print(mutation_score)
        print(mutation)        
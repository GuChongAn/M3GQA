import re
import os
import regex
import string
from collections import Counter
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, util

# metrics
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

# others
def is_image_openable(file_path):
    try:
        with Image.open(file_path) as img:
            return True
    except (IOError, SyntaxError) as e:
        return False
    
def get_image_summary(TEMP_PATH):
    question2image_summary = {}
    with open(os.path.join(TEMP_PATH, 'temp-imageSummary.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split("\t")
            question2image_summary[tmp[0]] = tmp[1]
    return question2image_summary


def get_topk_triples(triples, question, embedding_model, k = 10):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    triples_embedding = embedding_model.encode(triples, convert_to_tensor=True)
    similarity_scores = embedding_model.similarity(question_embedding, triples_embedding)[0]
    _, indices = torch.topk(similarity_scores, k=min(len(triples), k))
    return [triples[idx] for idx in indices]

def format_triples(triples):
    triples = re.split(r'\t|(?=<)', triples)
    pattern = r"<(.*?);(.*?);(.*?)>"
    results = []
    entity = set()
    relation = set()
    for triple in triples:
        match = re.match(pattern, triple)
        if match:
            e1, r, e2 = match.groups()
            results.append((e1.strip(), r.strip(), e2.strip()))
            entity.add(e1)
            entity.add(e2)
            relation.add(r)
    return results, entity, relation

def remove_repeat(triples, merged_entities):
    new_triples = set()
    for (e1, r, e2) in triples:
        for group in merged_entities:
            if e1 in group:
                e1 = group[0]
            if e2 in group:
                e2 = group[0]
        new_triples.add((e1, r, e2))
    return new_triples

def knowledge_graph_align(text_triples, table_triples, 
image_triples):
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            
        def find(self, i):
            if self.parent[i] != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]
        
        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                self.parent[root_j] = root_i
        
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_triples, text_entity, text_relation = format_triples(text_triples)
    if table_triples != "":
        table_triples, table_entity, table_relation = format_triples(table_triples)
    else:
        table_triples, table_entity, table_relation = [], set(), set()
    if image_triples != "":
        image_triples, image_entity, image_relation = format_triples(image_triples)
    else:
        image_triples, image_entity, image_relation = [], set(), set()

    all_entity_set = list(set.union(text_entity, table_entity, image_entity))
    all_entity = list(text_entity) + list(table_entity) + list(image_entity)
    
    # align
    embeddings = embeddings_model.encode(all_entity_set, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    n = len(all_entity_set)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] >= 0.75:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(all_entity_set[i])
    merged_entities = list(groups.values())

    text_triples = remove_repeat(text_triples, merged_entities)
    table_triples = remove_repeat(table_triples, merged_entities)
    image_triples = remove_repeat(image_triples, merged_entities)

    # cross-modal triples
    all_triples = list(text_triples) + list(table_triples) + list(image_triples)
    all_triples = set(all_triples)
    counter_entity = Counter(all_entity)
    rich_entity = set([e for (e, v) in counter_entity.items() if v > 1])
    rich_triples = []
    for rich_e in rich_entity:
        for (e1, r, e2) in all_triples:
            if rich_e == e1:
                rich_triples.add(e1)
            elif rich_e == e2:
                rich_triples.add(e1)

    return list(all_triples), list(rich_triples)

def candicate_triples_update(candicate_triples, select_triple, all_triples):
    select_entity_1, select_entity_2 = select_triple[0][0], select_triple[0][2]
    for triple in all_triples:
        entity = [triple[0], triple[2]]
        if select_entity_1 in entity or select_entity_2 in entity:
            candicate_triples.append(triple)
    return candicate_triples
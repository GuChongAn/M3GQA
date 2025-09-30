import os
import re
from tqdm import tqdm

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dataset.manymodalQA import ManymodalDataset
from torch.utils.data import DataLoader

from utils import get_image_summary, get_topk_triples, knowledge_graph_align, candicate_triples_update
from model.templateManager import TemplateManager
from sentence_transformers import SentenceTransformer

# Init 
TEMP_PATH = "./temp"
DATASET_PATH = "../../dataset/manymodalqa/"
L = 4

llm = ChatOpenAI(
    model='',
    base_url="",
    api_key=""
)
parser = StrOutputParser()
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# dataloader
multimodal_dataset = ManymodalDataset(DATASET_PATH)
multimodal_dataloader = DataLoader(dataset=multimodal_dataset)
question2image_summary = get_image_summary(TEMP_PATH)

# make temp dir
if not os.path.exists(TEMP_PATH):
    os.mkdir(TEMP_PATH)

# init langchain template
template_manager = TemplateManager()
text2graph_template = template_manager.get_template("text2graph_template")
table2graph_template = template_manager.get_template("table2graph_template")
reasoning_template = template_manager.get_template("reasoning_template")
answer_template = template_manager.get_template("answer_template")

# init langchain chain
text2graph_chain = (PromptTemplate.from_template(text2graph_template) | llm | parser)
table2graph_chain = (PromptTemplate.from_template(table2graph_template) | llm | parser)
reasoning_chain =  (PromptTemplate.from_template(reasoning_template) | llm | parser)
answer_chain = (PromptTemplate.from_template(answer_template) | llm | parser)

# answer
for data in tqdm(multimodal_dataloader):
    question = data['question'][0]
    texts = data['texts']
    tables = data['tables']
    
    # knowledge graph generate
    knowledge_graph_text, knowledge_graph_image, knowledge_graph_table = "", "", ""
    for text in texts:
        temp_text = text2graph_chain.invoke({'example': "", 'question': question, 'text': text['text'][0]}).strip()
        knowledge_graph_text += "\t".join(re.findall(r"<[^>]+>", temp_text))
    for table in tables:
        temp_table = table2graph_chain.invoke({'example': "", 'question': question, 'table': table['table'][0]}).strip()
        knowledge_graph_table += "\t".join(re.findall(r"<[^>]+>", temp_table))
    if question in question2image_summary:
        image_sumamry = question2image_summary[question]
        temp_image = text2graph_chain.invoke({'example': "", 'question': question, 'text': image_sumamry}).strip()
        knowledge_graph_image += "\t".join(re.findall(r"<[^>]+>", temp_image))
    
    # knowledge graph align
    all_triples, cross_triples = knowledge_graph_align(knowledge_graph_text, knowledge_graph_table, knowledge_graph_image)

    # init condicate_triples
    triples_topk = get_topk_triples(all_triples, question, embedding_model, 10)
    candicate_triples = triples_topk + cross_triples

    # generation reasoning path
    reasoning_path = ""
    for i in range(L):
        temp = reasoning_chain.invoke({"example": "", "question": question, "reasoning_path": reasoning_path, "triples": candicate_triples}).strip()
        select_triples = re.findall(r"<[^>]+>", temp)
        if len(select_triples) == 0:
            break
        candicate_triples = candicate_triples_update(candicate_triples, select_triples, all_triples)
        reasoning_path += (temp + "\t") 

    # answer question
    answer = answer_chain.invoke({'example': "", 'question': question, 'triples': reasoning_path})
    with open(os.path.join(TEMP_PATH, 'temp-answer.txt'), 'a+', encoding='utf-8') as f:
        f.write("{}\t{}\n".format(question, answer.strip().replace('\n', ' ').replace('\t', ' ')))
    break
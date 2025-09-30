import os
import json
from torch.utils.data import Dataset, DataLoader

class ManymodalDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data_list = self.MMQA_load()

    def __getitem__(self, id):
        # data item example: {
        #   'question': "xxxx",
        #   'answer': ["xxx", "xxxx"],
        #   'texts': [{"title":"xxxx", "text": "xxxx"},...],
        #   'images': [{"title":"xxxx", "path": "xxxx.jpg"},...],
        #   'tables’： [{"title":"xxxx", "table": "a | b | c \n d | e | f"},...]
        # }
        return self.data_list[id]

    def __len__(self):
        return len(self.data_list)
        
    def trans_table(self, table):
        table_list = [row.replace("\"", '').split(',') for row in table.split('\n')]
        table_str = '\n'.join([' | '.join(row) for row in table_list])
        return table_str

    def MMQA_load(self):
        with open(os.path.join(self.path, "dev.json"), 'r') as f:
            mmqa_dataset = json.load(f)
        mmqa_list = []
        for data in mmqa_dataset: 
            tmp = {}
            tmp['question'] = data['question']
            tmp['answers'] = [data['answer']]
            tmp['texts'] = [{'title': data['page'], 'text': data['text']}] 
            if data['image'] != None:
                tmp['images'] = [{'title': data['image']['caption'], 'path': data['id']+'.png'}]
            else:
                tmp['images'] = []
            if data['table'] != None:
                tmp['tables'] = [{'title': data['page'], 'table': self.trans_table(data['table'])}]
            else:
                tmp['tables'] = []
            mmqa_list.append(tmp)
        return mmqa_list
    
if __name__ == "__main__":
    manymodal_dataset = ManymodalDataset("../../../dataset/manymodalqa/")
    manymodal_dataloader = DataLoader(dataset=manymodal_dataset, batch_size=1)

    print(len(manymodal_dataloader))
    for data in manymodal_dataloader:
        print(data['question'])
        print(data['answers'])
        print(data['texts'])
        print(data['images'])
        print(data['tables'])
        break
        
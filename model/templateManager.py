class TemplateManager:
    def __init__(self):
        text2graph_template = "Given a text and a question, extract all the knowledge triples in the form of <head entity; relation; tail entity> that might be used to answer the question, "\
            "where head/tail entity is a phrase in the text and relation denotes a description of the relation between the head entity and the tail entity.\n\n" \
            "{example}\n" \
            "Question: {question} " \
            "Text: {text}\n" \
            "Knowledge Triples: "
        table2graph_template = "Given a question and a table,  extract all the knowledge triples in the form of <head entity; relation; tail entity> that might be used to answer the question, "\
            "where head/tail entity is a phrase in the table and relation denotes a description of the relation between the head entity and the tail entity.\n\n" \
            "{example}\n" \
            "Question: {question}\n" \
            "Table: {table}\n" \
            "Knowledge Triples: "
        reasoning_template = "Given a question, a known chain of reasoning, and some triples that might be helpful in answering the question, pick the most helpful triples from these triples\n\n" \
            "{example}\n" \
            "Question: {question}\n" \
            "Reasoning Chain: {reasoning_path}\n" \
            "Candicate Triples: {triples}" \
            "Selecte Triple:" 
        answer_template = "Given some konwledge graph triples and a question, please only output the answer to the question.\n "\
            "{example}\n" \
            "Triples: {triples}\n" \
            "Question: {question}\n" \
            "Answer: "
        image_summary_template = "Given a image and a question, summarize the image and focus on the parts that are relevant to the question.\n" \
            "Question: {question}"

        self.templates = {"text2graph_template": text2graph_template, 
                          "table2graph_template": table2graph_template,
                          "reasoning_template": reasoning_template,
                          "answer_template": answer_template,
                          "image_summary_template": image_summary_template}

    def get_template(self, name):
        return self.templates.get(name)

# 示例使用
if __name__ == "__main__":
    manager = TemplateManager()

    text2graph_template = manager.get_template("text2graph_template")

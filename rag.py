import os.path

import anthropic
from sentence_transformers import SentenceTransformer
import lancedb
import pandas as pd

os.environ["ANTHROPIC_API_KEY"] = open(os.path.expanduser("~/.anthropic/apikey"), "r").read().strip()

class SecurityRAG:
    def __init__(self):
        self.client = anthropic.Client()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.open_table("security_guidance")
        
    def get_relevant_context(self, query, num_docs=3):
        query_embedding = self.model.encode(query).tolist()
        results = self.table.search(query_embedding, vector_column_name="embedding") \
                           .limit(6) \
                           .to_pandas() \
                           .drop_duplicates(subset=['content']) \
                           .head(num_docs)
        
        context = "\n\n".join([
            f"URL: {row['url']}\nTitle: {row['title']}\nContent: {row['content']}"
            for _, row in results.iterrows()
        ])
        return context
    
    def get_response(self, query):
        context = self.get_relevant_context(query)
        
        system_prompt = """You are a security documentation assistant. Use the provided context to answer questions.
        If you cannot answer from the context, say so. Do not make up information."""
        
        messages = [
            {
                "role": "user",
                "content": f"""Context:
                {context}
                
                Question: {query}"""
            }
        ]
        
        response = self.client.messages.create(
            # see models: https://docs.anthropic.com/en/docs/resources/model-deprecations#model-status
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
            system=system_prompt,
            messages=messages
        )
        
        return response.content[0].text

# Usage example:
rag = SecurityRAG()
response = rag.get_response("What considerations are there for designing an IDAM system?")
print(response)
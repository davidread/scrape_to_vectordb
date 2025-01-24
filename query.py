import lancedb
from sentence_transformers import SentenceTransformer

# Connect to the database
db = lancedb.connect("./.lancedb")
table = db.open_table("security_guidance")

# Create embeddings for your query
model = SentenceTransformer('all-MiniLM-L6-v2')
query = "What are the password requirements?"
query_embedding = model.encode(query).tolist()

# Search
results = table.search(query_embedding, vector_column_name="embedding").limit(6).to_df().drop_duplicates(subset=['content']).head(3)
for _, result in results.iterrows():
    print(result.url)
    print(result)
    print()
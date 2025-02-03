#%%
from docling.document_converter import DocumentConverter
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo de linguagem natural
nlp = spacy.load("en_core_web_sm")

source_folder = "C:/Users/lojanovolar/Documents/Curriculos"
source_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(('.pdf', '.docx'))]
converter = DocumentConverter()

# Função para extrair texto dos documentos
def extract_text(file_path):
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

# Função de pré-processamento de texto
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Extrair e pré-processar textos dos currículos
documents = [preprocess_text(extract_text(file)) for file in source_files]

# Categorizar currículos
sales_keywords = ["comunicação", "honestidade", "agilidade", "proatividade", 'proativo', "vendas", "comercial", 'informática']
logistics_keywords = ["logística", "informática", "comunicação", "honestidade", "agilidade", "proatividade", 'proativo']
academic_keywords = ["bacharel", "licenciatura", "mestrado", "doutorado", "graduação", "pós-graduação", "universidade", "faculdade"]

# Convert all keywords to lowercase to ensure case insensitivity
sales_keywords = [keyword.lower() for keyword in sales_keywords]
logistics_keywords = [keyword.lower() for keyword in logistics_keywords]
academic_keywords = [keyword.lower() for keyword in academic_keywords]

def categorize_document(text):
    if any(keyword in text for keyword in academic_keywords):
        if any(keyword in text for keyword in sales_keywords):
            return "Sales (Academic)"
        elif any(keyword in text for keyword in logistics_keywords):
            return "Logistics (Academic)"
        else:
            return "Academic"
    else:
        if any(keyword in text for keyword in sales_keywords):
            return "Sales"
        elif any(keyword in text for keyword in logistics_keywords):
            return "Logistics"
        else:
            return "Uncategorized"

categories = [categorize_document(doc) for doc in documents]

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calcular similaridade de cosseno
similarity_matrix = cosine_similarity(tfidf_matrix)

# Rankear candidatos com base na similaridade
ranking = similarity_matrix.sum(axis=1).argsort()[::-1]

# Exibir ranking dos candidatos
for rank, index in enumerate(ranking):
    print(f"Rank {rank + 1}: {source_files[index]} - Category: {categories[index]}")

 # %%

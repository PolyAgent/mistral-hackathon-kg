"""Script to extract entities from a text file using the GLiNER model."""

import os
from pymongo import MongoClient
from gliner import GLiNER
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def get_paper_id(doc):
  return doc["doi"].split("/")[-1][6:]

def get_entity_pairs(doc, labels, model):
  abstract = doc["abstract"]
  entities = model.predict_entities(abstract, labels, threshold=0.5)
  entity_pairs = [{entity['label']: entity['text']} for entity in entities]
  entities = [entity['text'] for entity in entities]
  return entity_pairs, entities

# Create a connection to the MongoDB database
client = MongoClient(os.environ["MONGODB_URI"])
db = client.arxiv.papers_for_review

# available models: https://huggingface.co/urchade
model_name = "urchade/gliner_base"
ner_model = GLiNER.from_pretrained(model_name)
ner_model.eval()
print(f"Model {model_name} loaded.")

labels = ["Research Paper",
  "Author",
  "Conference",
  "Institution",
  "Field of Study",
  "Topic",
  "Technology",
  "Research Organization",
  "Subfield",
  "Year of Publication",
  "Competition",
  "Metric",
  "Preprint Repository",
  "Publication Process",
  "Research Goal",
  "Dataset",
  "Methodology",
  "Experiment",
  "Result",
  "Conclusion",
  "Limitation",
  "Future Work",
  "Collaboration",
  "Funding",
  "Affiliation",
  "Citation",
  "Reference",
  "Keyword",
  "Abstract",
  "Introduction",
  "Related Work",
  "Evaluation",
  "Performance",
  "Accuracy",
  "Precision",
  "Recall",
  "F1 Score",
  "Loss Function",
  "Optimization",
  "Hyperparameter",
  "Architecture",
  "Model",
  "Algorithm",
  "Code Repository",
  "Implementation",
  "Hardware",
  "Computational Resources",
  "Benchmark",
  "Ablation Study",
  "Comparative Analysis",
  "Visualization",
  "Interpretability",
  "Robustness",
  "Generalization",
  "Transfer Learning",
  "Domain Adaptation",
  "Multitask Learning",
  "Few-Shot Learning",
  "Zero-Shot Learning",
  "Unsupervised Learning",
  "Semi-Supervised Learning",
  "Self-Supervised Learning",
  "Contrastive Learning",
  "Adversarial Learning",
  "Federated Learning",
  "Continual Learning",
  "Lifelong Learning",
  "Meta-Learning",
  "Explainable AI",
  "Fairness",
  "Bias",
  "Privacy",
  "Security",
  "Ethics",
  "Societal Impact",
]

# Setup mistral api
mistral_prompt = """
Given the specified VOCABULARY for subjects and objects, the list of ENTITIES let's create the triplets from an abstract of a research paper. Keep triplets simple. Keep objects and subjects simple entities. Generate as many as you can (max 30). Output as json in such format. Don't use tools:

{"subject": "The Transformer", "predicate": "is topic", "object": "Research Paper"}
{"subject": "The Transformer", "predicate": "uses", "object": "large training data"},
"""
api_key = os.environ["MISTRAL_API_KEY"]

client = MistralClient(api_key=api_key)

ner_update_list = []
for doc in db.find():
    print(doc["title"])
    name_entity_pairs, entities = get_entity_pairs(doc, labels=labels, model=ner_model)
    db.update_one({"_id": doc["_id"], "named_entities": name_entity_pairs})
    user_message = f"""ABSTRACT:
    
    {doc["abstract"]}
    
    ENTITIES:
    
    {entities}
    
    VOCABULARY:
    
    {labels}
    """
    messages = [
    ChatMessage(role="system", content=mistral_prompt),
    ChatMessage(role="user", content = user_message)
    ]

    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages,
    )
    extracted_triples = chat_response.choices[0].message.content
    db.update_one({"_id": doc["_id"]}, {"$set": {"triples": extracted_triples}})

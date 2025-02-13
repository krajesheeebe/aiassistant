import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pymongo import MongoClient

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context and FamilyLinkingDetails data and InvitationDetails data:

Context:
{context}

FamilyLinkingDetails Data:
{family_data}

InvitationDetails Data:
{invitation_data}
---

Answer the question based on the above context: {question}
"""

def get_db_invitation_data(customer_id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["usrsp"]
    collection = db["invitationDetails"]
    # query if "inviterGcifId" or "inviteeGcifId" is customer_id
    invitation_data_cursor = collection.find({"$or": [{"inviterGcifId": customer_id}, {"inviteeGcifId": customer_id}]})
    return list(invitation_data_cursor)

def get_db_family_data(customer_id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["usrsp"]
    collection = db["familyLinkingDetails"]
    # query if "familyMembers.inviterGcifId" or "familyMembers.inviteeGcifId" is customer_id
    family_data_cursor = collection.find({"$or": [{"familyMembers.inviterGcifId": customer_id}, {"familyMembers.inviteeGcifId": customer_id}]})
    return list(family_data_cursor)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("customer_id", type=str, help="The customer ID.")
    args = parser.parse_args()
    query_text = args.query_text
    customer_id = args.customer_id
    query_rag(query_text, customer_id)


def query_rag(query_text: str, customer_id: str):
    # Get customer data from MongoDB.
    db_invitation_data = get_db_invitation_data(customer_id)
    db_family_data = get_db_family_data(customer_id)
    print(f"Invitation data: {db_invitation_data}")
    print(f"Family data: {db_family_data}")
    if not db_family_data and not db_invitation_data:
        print(f"No data found for customer ID: {customer_id}")
        return
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, invitation_data=db_invitation_data, family_data=db_family_data, question=query_text)
    print(prompt)

    model = OllamaLLM(model="llama2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

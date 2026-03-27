import openai
from pinecone import Pinecone
import streamlit as st

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = client
        self.model = "gpt-4.1-nano"
        self.prompt = (
            "You are a content moderator. Your task is to determine if the user's input contains any of the following:\n"
            "1. Rude or disrespectful language.\n"
            "2. Hate speech or discrimination.\n"
            "3. Malicious provocation.\n\n"
            "If the input contains any of the above, respond ONLY with 'Yes'. Otherwise, respond ONLY with 'No'."
        )        

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Obnoxious_Agent
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        return "yes" in response.strip().lower()

    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=3
            )
            
        raw_content = response.choices[0].message.content
        
        return self.extract_action(raw_content)


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        # TODO: Initialize the Context_Rewriter agent
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.system_prompt = (
            "You are a Context Resolver. Your only job is to rewrite the 'Latest Query' "
            "to be a standalone question by replacing pronouns (like 'it', 'they', 'this', 'that') "
            "with the actual subject from the 'Chat History'.\n\n"
            "Example:\n"
            "Chat History: User: What is SVM? Assistant: SVM is a model...\n"
            "Latest Query: Tell me more about it.\n"
            "Rewritten: Tell me more about Support Vector Machine (SVM).\n\n"
            "ONLY return the rewritten question. If no change is needed, return the original."
        )

    def rephrase(self, user_history, latest_query):
        # TODO: Resolve ambiguities in the final prompt for multiturn situations
        history_context = "\n".join(user_history[-3:])
        
        full_input = f"Chat History:\n{history_context}\n\nLatest Query: {latest_query}"
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_input}
                ],
                temperature=0
            )
            
        return response.choices[0].message.content.strip()


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        # TODO: Initialize the Query_Agent agent
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings_model = embeddings
        self.model = "gpt-4.1-nano"
        #check if db query needed
        self.prompt = (
            "You are a routing assistant. Determine if the following user query requires "
            "external knowledge from a textbook about Machine Learning. "
            "If it is a technical question or requires specific info, respond 'Search'. "
            "If it is a greeting or general chat, respond 'General'."
        )

    def query_vector_store(self, query, k=5):
        #query embedding
        res = self.client.embeddings.create(
            input=[query],
            model=self.embeddings_model
        )
        query_vector = res.data[0].embedding

        #Pinecone
        search_results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )
        
        contexts = [item['metadata']['text'] for item in search_results['matches']]
        return "\n".join(contexts)

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Query_Agent agent
        self.prompt = prompt

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        if "search" in response.lower():
            return self.query_vector_store(query)
        else:
            return ""

    def check_relevance(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        
        # Let AI judge to search or not
        return self.extract_action(response.choices[0].message.content, query)


class Answering_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Answering_Agent
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.system_prompt = (
            "You are a knowledgeable Machine Learning Assistant. "
            "Use the provided 'Context' (retrieved from a textbook) and the 'Conversation History' "
            "to answer the user's latest query accurately. "
            "If the context is not relevant or empty, rely on your general knowledge but mention it. "
            "Keep your tone professional, helpful, and concise."
        )

    def generate_response(self, query, docs, conv_history, k=5):
        # TODO: Generate a response to the user's query
        context_str = "\n".join(docs) if isinstance(docs, list) else docs
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conv_history[-k:]])

        user_message = (
            f"Context from textbook:\n{context_str}\n\n"
            f"Conversation History:\n{history_str}\n\n"
            f"User's latest query: {query}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
        )
    
        return response.choices[0].message.content.strip()


class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt = (
            "You are a relevance evaluator. You will be given a user query and a set of retrieved document segments. "
            "Your task is to determine if the documents contain information that can help answer the query. "
            "Respond ONLY with 'Relevant' if the documents are useful, "
            "or 'Irrelevant' if they are not related to the query."
        )

    def get_relevance(self, query, retrieved_docs) -> str:
        # TODO: Get if the returned documents are relevant
        if not retrieved_docs:
            return "Irrelevant"

        docs_text = "\n".join(retrieved_docs) if isinstance(retrieved_docs, list) else retrieved_docs
        conversation_input = f"User Query: {query}\n\nRetrieved Documents:\n{docs_text}"

        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": conversation_input}
                ],
                temperature=0
            )
            
        # 'Relevant' or 'Irrelevant'
        status = response.choices[0].message.content.strip()
        return status


class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        # TODO: Initialize the Head_Agent
        # openai
        self.openai_key = openai_key
        self.client = openai.OpenAI(api_key=openai_key)
        
        # pinecone
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(pinecone_index_name)
        
        self.setup_sub_agents()
        
        self.chat_history = []

    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.rewriter_agent = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.client, "text-embedding-3-small")
        self.relevance_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def run(self, user_input):
        # TODO: Run the main loop for the chatbot
        # Obnoxious Agent
        if not self.obnoxious_agent.check_query(user_input):
            # Context Rewriter
            history_list = [f"{m['role']}: {m['content']}" for m in self.chat_history]
            optimized_query = self.rewriter_agent.rephrase(history_list, user_input)

            st.write("🔍 Analying if need to search Pinecone Database ...")
    
            # Query Agent
            retrieved_docs = self.query_agent.check_relevance(optimized_query)
    
            # Relevant Documents Agent
            final_context = ""
            if retrieved_docs:
                st.write("📖 Find results from Pinecone, evaulating correctness...")
                relevance_status = self.relevance_agent.get_relevance(optimized_query, retrieved_docs)
                if relevance_status == "Relevant":
                    final_context = retrieved_docs
            else:
                st.write("💡 No need to search Pinecone")
    
            # Answering Agent
            response = self.answering_agent.generate_response(
                query = optimized_query,
                docs = final_context,
                conv_history = self.chat_history
            )

            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
        else:
            response = "Obnoxious content detected, Plz be nice"

        return response
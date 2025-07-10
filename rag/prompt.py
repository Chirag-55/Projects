def build_prompt(contexts, question):
    if contexts:
        context_text = "\n---\n".join(contexts)
        return f"""You are a helpful and knowledgeable medical assistant.

Use the following document context if it helps answer the question. If the answer is not clearly found in the context, answer using your own medical knowledge.

Context:
{context_text}

Question: {question}
Answer:"""
    else:
        return f"""You are a helpful and knowledgeable medical assistant.

Please answer the following medical question to the best of your ability using your own knowledge.

Question: {question}
Answer:"""

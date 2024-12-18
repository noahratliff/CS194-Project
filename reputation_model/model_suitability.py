# model_suitability.py
import numpy as np
from sentence_transformers import SentenceTransformer


class ModelSuitability:
    def __init__(self, agent_names, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.question_embeddings = []
        self.agent_names = agent_names
        self.successes = {a: [] for a in agent_names}

    def update(self, question, successes):
        for agent_name in successes:
            success = successes[agent_name]
            self.successes[agent_name].append(1 if success else 0)

        # Use convert_to_tensor=False to get NumPy arrays
        embedding = self.model.encode(question, convert_to_tensor=False)
        self.question_embeddings.append(embedding)

    def compute_similarities(self, new_question_embedding):
        # Compute cosine similarities between new question and all previous questions
        similarities = np.array(
            [
                np.dot(new_question_embedding, emb)
                / (np.linalg.norm(new_question_embedding) * np.linalg.norm(emb))
                for emb in self.question_embeddings
            ]
        )
        # Normalize similarities from [-1, 1] to [0, 1]
        # similarities = (similarities + 1) / 2
        return similarities

    def suitability_score(self, new_question):
        new_question_embedding = self.model.encode(
            new_question, convert_to_tensor=False
        )
        similarities = self.compute_similarities(new_question_embedding)

        if len(similarities) == 0:
            return 0.0  # No data to base suitability on


        agent_suitabilities = {}
        for agent_name in self.agent_names:
            successes = np.array(self.successes[agent_name])

            # Calculate weighted success rate
            weighted_successes = similarities * successes
            weighted_total = similarities.sum()
            # print(f"{similarities=}, {successes=}, {weighted_successes=} \n \
            #       {weighted_total=}, {weighted_successes.sum()/weighted_total=}")

            if weighted_total == 0:
                agent_suitabilities[agent_name] = 0 # Avoid division by zero

            suitability = weighted_successes.sum() / weighted_total

            # Ensure suitability is between 0 and 1
            suitability = max(0.0, min(1.0, suitability))

            agent_suitabilities[agent_name] = suitability
        
        return agent_suitabilities

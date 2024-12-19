from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
import os
import random

from gpt_utils import query_gpt4
import time
from datasets import load_dataset

from reputation_model.model_suitability import ModelSuitability

# external API for updating reputation scores
def update_reputation(question, agent_correctness):
    pass


def get_reputation_context(agent_name, reputation_scores):
    pass

class System():
    def initialize_group_chat(self, use_reputation=False):
        agent_model_types = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
        # Initialize agents
        llm_config = {
            "config_list": [
                {"model": agent_model_types[0], "api_key": os.environ.get("OPENAI_API_KEY")}
            ]
        }
        
        self.num_agents = 4
        self.agents = [
            ConversableAgent(
                name=f"Facilitator",
                llm_config=llm_config,
                system_message="""You are a facilitator. However, do not provide any input, answer, or analysis to the given question. Your only job is to ask for the final answer when the agents are done discussing. DO NOT provide your own answers to the questions. Let them discuss for two rounds and then simply ask "Can each agent provide their final answer? Give just the answer please (an integer and nothing else)." when the other agents' conversation is over. After the first round, encourage them to check their answers and/or keep discussing, but still DO NOT provide any explanation or help with the answer.""",
                silent=False,
            )
        ]

        for i in range(1, self.num_agents + 1):
            agent = ConversableAgent(
                name=f"Agent{i}",
                system_message="""
                                You are a collaborative agent. After each given question, you will engage in multi-agent discussion.
                                During the conversation portion, explain your reasoning and answer. You will be given information about how other agents previously performed on similar questions, 
                                which you should use to decide how much to trust them.
                                When you are asked for your final answer, you will vote for your final answer and give your answer as only the number with no explanation. Your answer should only be one of the following numbers: 0, 1, 2, or 3. Do not send any other text or any other explanation.
                                It does not have to be the same answer as the other agents.
                                """,
                llm_config={
                    "config_list": [
                        {"model": agent_model_types[i], "api_key": os.environ["OPENAI_API_KEY"]}
                    ]
                },
                # human_input_mode="NEVER",
                silent=False,
            )

            self.agents.append(agent)
        

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=(self.num_agents + 1) * 2,
            speaker_selection_method="round_robin",
        )

        self.chat_manager = GroupChatManager(self.group_chat)

        self.non_facilitator_agents = self.agents[1:]
        self.non_facilitator_agent_names = [agent.name for agent in self.agents[1:]]
        self.model_suitability = ModelSuitability(agent_names = self.non_facilitator_agent_names)

    def process_question(self, question, correct_answer):
        # PRELIMINARY ANSWERS 
        initial_responses = {}
        print(f"{question=}")
        print(f"{correct_answer=}")

        # Initial Question
        for agent in self.non_facilitator_agents:
            initial_prompt = f"Question: {question}. Can each agent provide an initial answer to the question? Your output should only be a single integer: 0, 1, 2, or 3. Do not provide your answer in the form of a sentence, just as a single number. For example, if you believe the correct answer is choice 2, just respond with '2' and nothing else."
            initial_answer = agent.generate_reply(messages=[{"content": initial_prompt, "role": "user"}])
            initial_responses[agent.name] = initial_answer
            
            print(f"Agent {agent.name} Initial Answer: {initial_answer}")

        agent_reputations = self.model_suitability.suitability_score(question)
        print(f"{agent_reputations=}")
        # if agent_reputations == 0.0:
        #     agent_reputations = "N/A"
        # print(f"Agents' Initial Answer Correctness on Similar Questions: {agent_reputations}")


        # Facilitator posts the agents' reputation for context.
        self.chat_manager.run_chat(
            messages=[{"role": "user", "content": f"Question: {question} \n Agents' Past Performance on Simliar Problems: {agent_reputations}. Begin discussion:"}], # {question}\n Initial Answers: {initial_responses}
            config=self.group_chat,
            sender=self.group_chat.agents[0],
        )
                
        final_answers = {}
        for message in self.group_chat.messages[-self.num_agents:]:
            name = message["name"]
            final_answers[name] = message["content"].strip().lower()
            print(f"{name}'s final answer: {final_answers[name]}")
        
        # TODO: have players vote now, conditioned on the chat dialogue
        print(f'{final_answers=}')
        print(f'{correct_answer=}')
        # calculate votes 
        vote_values = list(final_answers.values())
        final_answer = max(set(vote_values), key=vote_values.count)
        
        # evaluate entire system
        is_correct = final_answer == str(correct_answer)
        print(
            f"The group's final answer was {'correct' if is_correct else 'incorrect'}"
        )

        final_successes = {}
        for agent_name in self.non_facilitator_agent_names:
            # print(f"{final_answers[agent_name]=}", )
            final_successes[agent_name] = (str(final_answers[agent_name]) == str(correct_answer))
            print(f"final_successes[{agent_name}]={final_successes[agent_name]}")

        self.model_suitability.update(question, successes = final_successes)
        
        return is_correct
    
    
    # Function to evaluate GPT-4 on a dataset subset
    def evaluate_gpt4_on_mmlu(self, subject="high_school_mathematics", sample_size=20):
        num_correct = 0
        total = 0
        
        # Load the MMLU dataset
        dataset = load_dataset("cais/mmlu", subject, split="test")

        # Sample a subset of questions
        sampled_questions = random.sample(list(dataset), sample_size)

        correct_count = 0
        for idx, question in enumerate(sampled_questions):
            prompt = f"""Question: {question['question']}\nOptions:\n0: {question['choices'][0]}\n1: {question['choices'][1]}\n2: {question['choices'][2]}\n3: {question['choices'][3]}\nAnswer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not answer in the form a sentence; answer only with a single number as your output."""
            
            print(f"Processing question {idx+1}/{sample_size}...")
            # print(question)
            
            is_correct = self.process_question(prompt, question["answer"])
            
            if is_correct:
                num_correct += 1
            total += 1
            
                        
            # gpt4_answer = query_gpt4(prompt)

            # Compare GPT-4's answer to the ground truth
            # print(gpt4_answer)
            # print(question["answer"])

            # if gpt4_answer == str(question['answer']):
            #     correct_count += 1
            # else:
            #     print(f"Wrong Answer: {gpt4_answer} (Correct: {question['answer']})")
        
        # Calculate accuracy
        accuracy = (num_correct / total) * 100
        print(f"\nAccuracy on {subject}: {accuracy:.2f}%")
        return accuracy
    
    def evaluate_gpt4_on_mmlu_mixed(self, subjects=["college_medicine", "international_law"], sample_size=20):
        num_correct = 0
        total = 0
        
        # Load the MMLU dataset
        datasets = []
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            datasets.append(dataset)

        # Sample a subset of questions
        sampled_questions = []
        for dataset in datasets:
            sampled_questions += (random.sample(list(dataset), sample_size//len(subjects)))
        random.shuffle(sampled_questions)

        correct_count = 0
        for idx, question in enumerate(sampled_questions):
            print(f"{question=}")
            prompt = f"""Question: {question['question']}\nOptions:\n0: {question['choices'][0]}\n1: {question['choices'][1]}\n2: {question['choices'][2]}\n3: {question['choices'][3]}\n""" # Answer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not answer in the form a sentence; answer only with a single number as your output."""
            
            print(f"Processing question {idx+1}/{sample_size}...")
            # print(question)
            
            is_correct = self.process_question(prompt, question["answer"])
            
            if is_correct:
                num_correct += 1
            total += 1
            
                        
            # gpt4_answer = query_gpt4(prompt)

            # Compare GPT-4's answer to the ground truth
            # print(gpt4_answer)
            # print(question["answer"])

            # if gpt4_answer == str(question['answer']):
            #     correct_count += 1
            # else:
            #     print(f"Wrong Answer: {gpt4_answer} (Correct: {question['answer']})")
        
        # Calculate accuracy
        accuracy = (num_correct / total) * 100
        print(f"\n Total Accuracy: {accuracy:.2f}%")
        return accuracy


    # def evaluate_system(self):
           
    #     if is_correct:
    #         correct += 1
    #     total_questions += 1
    #     # update reputation scores


    #     questions = [
    #         {"question": "What is the capital of France?", "answer": "Paris"},
    #         {"question": "What is 2 + 2?", "answer": "4"},
    #         {"question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"},
    #     ]

    #     reputation_scores = []

    #     for qa in questions:
    #         question = qa["question"]
    #         correct_answer = qa["answer"].lower()
    #         print(f"\nQuestion: {question}")
    #         chat_manager.run_chat(
    #             messages=[{"role": "user", "content": question}],
    #             config=group_chat,
    #             sender=agents[0],
    #         )

    #         # Get the final answer from the agents

    #         final_answers = {}
    #         for message in group_chat.messages[-num_agents:]:
    #             name = message["name"]
    #             final_answers[name] = message["content"].strip().lower()
    #             print(f"{name}'s final answer: {final_answers[name]}")

    #         vote_values = list(final_answers.values())
    #         final_answer = max(set(vote_values), key=vote_values.count)

    #         is_correct = final_answer == correct_answer
    #         print(
    #             f"The group's final answer was {'correct' if is_correct else 'incorrect'}"
    #         )

    #         for agent in agents:
    #             if agent.name == "Facilitator":
    #                 continue
    #             is_correct = final_answers[agent.name] == correct_answer
    #             print(
    #                 f"{agent.name}'s final answer was {'correct' if is_correct else 'incorrect'}"
    #             )

            # TODO: Update reputation scores

            # # Each agent gives their final answer
            # final_answers = {}
            # for agent in agents:
            #     name = agent.name
            #     reputation_context = (
            #         get_reputation_context(name, reputation_scores)
            #         if use_reputation
            #         else ""
            #     )
            #     agent_message = f"{reputation_context}\n\nPlease provide your final answer to the question."
            #     response = agent.generate_reply(
            #         messages=[{"role": "user", "content": agent_message}]
            #     )
            #     print(response)
            # final_answers[name] = response.content.strip()
            # print(f"{name}'s final answer: {final_answers[name]}")
            # time.sleep(1)

            #     # Agents vote on the final answer (collectively determine the most common final answer)
            #     vote_values = list(final_answers.values())
            #     final_answer = max(set(vote_values), key=vote_values.count)
            #     print(f"\nFinal answer after voting: {final_answer}")

            #     # Check if each agent's final answer was correct
            #     for name in agent_names:
            #         is_correct = final_answers[name].lower() == correct_answer.lower()
            #         reputation = update_reputation(
            #             name, question, is_correct, reputation_scores
            #         )
            #         print(
            #             f"{name}'s final answer was {'correct' if is_correct else 'incorrect'}. Updated reputation: {reputation}"
            #         )

            #     # Pass the correct answer to agents (for future context)
            #     for name, agent in agents.items():
            #         agent.receive_message(
            #             sender="System",
            #             message=f"The correct answer was: {correct_answer}",
            #             silent=True,
            #         )


        # print("\nFinal reputation scores:")
        # for name, score in reputation_scores.items():
        #     print(f"{name}: {score}")


if __name__ == "__main__":

    system = System()
    system.initialize_group_chat()
    # system.evaluate_gpt4_on_mmlu(sample_size=5)
    system.evaluate_gpt4_on_mmlu_mixed(sample_size=3)

    # TODO: try using these as agents, since they have different expertise levels now
    # agents = [
    #     ConversableAgent(
    #         name=f"gpt-4",
    #         system_message='''You are a collaborative agent that will engage in constructive discussions to come to a collective answer.
    #                         You will be given information about how other agents performed on each question, 
    #                         which you should use to decide how much to trust their information.''',
    #         llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
    #         # human_input_mode="NEVER",
    #         silent = False
    #     ),
    #     ConversableAgent(
    #         name=f"gpt-4o-mini",
    #         system_message='''You are a collaborative agent that will engage in constructive discussions to come to a collective answer.
    #                         You will be given information about how other agents performed on each question, 
    #                         which you should use to decide how much to trust their information.''',
    #         llm_config={"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]},
    #         # human_input_mode="NEVER",
    #         silent = False
    #     ),
    #     ConversableAgent(
    #         name=f"gpt-3",
    #         system_message='''You are a collaborative agent that will engage in constructive discussions to come to a collective answer.
    #                         You will be given information about how other agents performed on each question, 
    #                         which you should use to decide how much to trust their information.''',
    #         llm_config={"config_list": [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]},
    #         # human_input_mode="NEVER",
    #         silent = False
    #     )
    # ]
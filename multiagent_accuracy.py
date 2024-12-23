from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
import os
import random

from gpt_utils import query_gpt4
import time
from datasets import load_dataset
import openai
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import re




from reputation_model.model_suitability import NaiveModelSuitability

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_ORG_ID")
client = openai.OpenAI()


# external API for updating reputation scores
def update_reputation(question, agent_correctness):
    pass


def get_reputation_context(agent_name, reputation_scores):
    pass

class DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class System:
    def initialize_results_csv(self):
        # Create results folder if it doesn't exist
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        # Initialize the CSV with the appropriate columns
        columns = []

        columns.extend(["question", "correct_answer"])

        for i in range(1, self.num_agents + 1):
            columns.append(f"Agent{i}_initial_response")
            columns.append(f"Agent{i}_conversational_response")

        # Add additional metadata columns if needed
        

        if not os.path.exists(self.results_file):
            pd.DataFrame(columns=columns).to_csv(self.results_file, index=False)

    def save_results(self, question, correct_answer, initial_answers, final_answers, reputation_scores, num_files, new_file=True):
        print("EHLLLOOOOOOOOOOOOOOOOOOOOOOOOOO")
        if not new_file:
            # Load existing results
            results_df = pd.read_csv(self.results_file)
        else:
            filename = self.results_file.replace(".csv", f"{num_files}.csv")
            try:
                results_df = pd.read_csv(filename)
            except FileNotFoundError:
                results_df = pd.DataFrame()

        # Prepare a new row for the current question
        row = {}
        for agent_name, initial_answer in initial_answers.items():
            row[f"{agent_name}_initial_response"] = initial_answer

        for agent_name, final_answer in final_answers.items():
            row[f"{agent_name}_conversational_response"] = final_answer

        for agent_name, reputation in reputation_scores.items():
            row[f"{agent_name}_reputation_score"] = reputation

        # Add metadata columns
        row["question"] = question.replace("\n", " ")
        row["correct_answer"] = correct_answer
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

        if not new_file:
            print("CONCATENATING RESULTS IN SELF.RESULTS_FILE")
            # Append the new row to the DataFrame and save it
            results_df.to_csv(self.results_file, index=False)
        else:
            # create new csv to save results in
            print("SAVING RESULTS IN: ", filename)
            results_df.to_csv(filename, index=False)

    def initialize_group_chat(self, use_reputation=False, reputation_method="cosine"):
        agent_model_types = [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            # "gpt-3.5-turbo",
            "gpt-4o-mini",
        ]
        # Initialize agents
        llm_config = {
            "config_list": [
                {
                    "model": agent_model_types[0],
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                }
            ]
        }

        self.num_agents = len(agent_model_types) - 1
        self.agents = [
            ConversableAgent(
                name=f"Facilitator",
                llm_config=llm_config,
                system_message="""You are a facilitator. However, do not provide any input, answer, or analysis to the given question. Your only job is to ask for the final answer when the agents are done discussing. DO NOT provide your own answers to the questions. Let them discuss for two rounds and then simply ask "Can you provide your final answer? Give just the answer in the form '0', '1', '2', or '3'. You shouldn't guess what other agents' answers are, just report your final answer as one of the following numbers: 0, 1, 2, or 3. Do not send any other text or any other explanation." """,
                silent=False,
                human_input_mode="NEVER"
            )
        ]

        for i in range(1, self.num_agents + 1):
            agent = ConversableAgent(
                name=f"Agent{i}",
                system_message="""
                                You are a collaborative agent. After each given question, you will engage in multi-agent discussion.
                                During the conversation portion, explain your reasoning and answer. You will be given information about how other agents previously performed on similar questions, 
                                which you should use to decide how much to trust them. Don't be wordy.
                                When you are asked for your final answer, you will vote for your final answer and give your answer as only the number with no explanation. Your answer should only be one of the following numbers: 0, 1, 2, or 3. Do not send any other text or any other explanation.
                                It does not have to be the same answer as the other agents.
                                """,
                llm_config={
                    "config_list": [
                        {
                            "model": agent_model_types[i],
                            "api_key": os.environ["OPENAI_API_KEY"],
                        }
                    ]
                },
                human_input_mode="NEVER",
                silent=False,
            )

            self.agents.append(agent)

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=(self.num_agents + 1) * 1 + 1,
            speaker_selection_method="round_robin",
        )

        self.chat_manager = GroupChatManager(self.group_chat)

        self.non_facilitator_agents = self.agents[1:]
        self.non_facilitator_agent_names = [agent.name for agent in self.agents[1:]]
        self.model_suitability = NaiveModelSuitability(
            agent_names=self.non_facilitator_agent_names
        )

        assert reputation_method in ["cosine", "accuracy", "baseline"]
        self.reputation_method = reputation_method

        self.results_folder = "statistics/normal_accuracy"
        self.results_file = os.path.join(self.results_folder, "evaluation_results_with_rep.csv")
        self.initialize_results_csv()

    def process_question(self, question, correct_answer, num_files):
        # reset group chat before every question
        self.group_chat.reset()

        # clear agent chat history before every question
        for agent in self.agents:
            agent.clear_history()
        
        # PRELIMINARY ANSWERS
        initial_answers = {}
        print(f"{question=}")
        print(f"{correct_answer=}")

        # Initial Question TODO: potentially fi
        initial_prompt = f"{question}. Can you provide an initial answer to the question? Your output should only be a single integer: 0, 1, 2, or 3. Do not provide your answer in the form of a sentence, just as a single number. For example, if you believe the correct answer is choice 2, just respond with '2' and nothing else. You don't care about other agents' answers at this point, and don't guess what other agents' answers."
        print(f"Initial prompt: {initial_prompt}")

        for agent in self.non_facilitator_agents:
            initial_answer = agent.generate_reply(
                messages=[{"content": initial_prompt, "role": "user"}]
            )
            if initial_answer.isnumeric():
                initial_answers[agent.name] = initial_answer
            else:
                # if initial answer is in wrong format (ex: agent2: my answer is 1, just get final number)
                match = re.findall(r'\d+', initial_answer)
                last_number = match[-1] if match else None
                initial_answers[agent.name] = last_number
                print(f"PARSED NONNUMERIC AGENT ANSWER TO BE {initial_answers[agent.name]}")

            print(f"{agent.name} Initial Answer: {initial_answer}")

        agent_reputations = self.model_suitability.suitability_score(question)
        print(f"{agent_reputations=}")
        # if agent_reputations == 0.0:
        #     agent_reputations = "N/A"
        # print(f"Agents' Initial Answer Correctness on Similar Questions: {agent_reputations}")

        # Facilitator posts the agents' reputation for context.
        if self.reputation_method == "baseline":
            # facilitator_prompt =
            pass
        else:
            facilitator_prompt = f"{question} \n Agents' Past Performance on Simliar Problems (between 0 and 1): {agent_reputations}. Begin discussion:"

        self.chat_manager.run_chat(
            messages=[
                {
                    "role": "user",
                    "content": facilitator_prompt,
                }
            ],  # {question}\n Initial Answers: {initial_responses}
            config=self.group_chat,
            sender=self.group_chat.agents[0],
        )

        final_answers = {}

        for agent in self.non_facilitator_agents:
            response = agent.generate_reply(self.group_chat.messages)
            if response.isnumeric():
                final_answers[agent.name] = response
            else:
                match = re.findall(r'\d+', initial_answer)
                last_number = match[-1] if match else None
                final_answers[agent.name] = last_number
                print(f"PARSED NONNUMERIC FINAL ANSWER TO BE {final_answers[agent.name]}")


        # TODO: have players vote now, conditioned on the chat dialogue
        print(f"{final_answers=}")
        print(f"{correct_answer=}")

        # calculate votes
        vote_values = list(final_answers.values())
        final_answer = max(set(vote_values), key=vote_values.count)

        # evaluate entire system
        is_correct = final_answer == str(correct_answer)
        print(
            f"The group's final answer was {'correct' if is_correct else 'incorrect'}"
        )

        initial_successes = {}
        for agent_name in self.non_facilitator_agent_names:
            # print(f"{final_answers[agent_name]=}", )
            initial_successes[agent_name] = str(initial_answers[agent_name]) == str(
                correct_answer
            )
            print(f"initial_successes[{agent_name}]={initial_successes[agent_name]}")

        # final_successes = {}
        # for agent_name in self.non_facilitator_agent_names:
        #     # print(f"{final_answers[agent_name]=}", )
        #     final_successes[agent_name] = str(final_answers[agent_name]) == str(
        #         correct_answer
        #     )
        #     print(f"final_successes[{agent_name}]={final_successes[agent_name]}")

        self.model_suitability.update(
            question, successes=initial_successes
        )  # TODO: reputation based on answer before or after discussion?

        # # write results for each question to save_progress_multiagent_system.csv
        # save_line = {
        #     "question": question,
        #     "final_answer": final_answer,
        #     "correct_answer": correct_answer,
        #     "is_correct": is_correct,
        # }
        # # for agent, initial in initial_answers.values():
        # #     save_line[f"{agent}_initial_answer"] = initial
        # # for agent, final in final_answers.values():
        # #     save_line[f"{agent}_final_answer"] = final

        # with open('result/save_progress_multiagent_system.csv', mode='w', newline='') as file:

        self.save_results(
            question, correct_answer, initial_answers, final_answers, agent_reputations, num_files
        )

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
            # prompt = f"""Question: {question['question']}\nOptions:\n0: {question['choices'][0]}\n1: {question['choices'][1]}\n2: {question['choices'][2]}\n3: {question['choices'][3]}\nAnswer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not answer in the form a sentence; answer only with a single number as your output."""
            prompt = f"""Question: {question['question']}\nOptions:\n0: {question['choices'][0]}\n1: {question['choices'][1]}\n2: {question['choices'][2]}\n3: {question['choices'][3]}\n"""

            print(f"Processing question {idx+1}/{sample_size}...")
            # print(question)

            is_correct = self.process_question(prompt, question["answer"], 0)

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

    def evaluate_gpt4_on_mmlu_mixed(
        self, subjects=["college_medicine", "international_law"], sample_size=20
    ):
        num_correct = 0
        total = 0

        # Load the MMLU dataset
        datasets = []
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            datasets.append(dataset)

        # Sample a subset of questions
        sampled_questions = []
        # TODO: change later
        for dataset in datasets:
            sampled_questions += random.sample(
                list(dataset), sample_size // len(subjects)
            )
        random.shuffle(sampled_questions)

        for idx, qa in enumerate(sampled_questions):
            # print(f"{question=}")
            self.initialize_group_chat()
            question = f"""Question: {qa['question']}\nOptions:\n0: {qa['choices'][0]}\n1: {qa['choices'][1]}\n2: {qa['choices'][2]}\n3: {qa['choices'][3]}\n"""  # Answer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not answer in the form a sentence; answer only with a single number as your output."""

            print(f"Processing question {idx+1}/{sample_size}...")
            # print(question)

            is_correct = self.process_question(question, qa["answer"], 0)

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


    def full_eval_mmlu_mixed(self, subjects=["college_mathematics", "abstract_algebra", "college_physics"], batch_size = 16, max_iters = None):
        all_num_correct = 0
        all_total = 0
        num_result_files = len([f for f in os.listdir(self.results_folder) if os.path.isfile(os.path.join(self.results_folder, f))])
        
        # Load the MMLU dataset
        dataset_list = []
        for subject in subjects:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            print(f"initial {type(dataset)=}")
            dataset_list += list(dataset)
        
        dataset = DictDataset(dataset_list)
        def collate_fn(batch): return batch
        print(f"{type(dataset_list)=}, {type(dataset_list[0])=} {len(dataset_list)=}")
        
        # concat_datasets = torch.utils.data.ConcatDataset(dataset_list)        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        batch_accuracies = []
        running_accuracies = []
        num_epochs = 3
        for epoch in range(num_epochs):
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            for batch_i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                batch_num_correct = 0
                batch_total = 0
                
                if max_iters is not None and batch_i >= max_iters:
                    break
                
                print(f"{type(batch)=}, {len(batch)=}")
                for idx, qa in enumerate(batch):
                    print(f"{qa=}")
                    question = f"""Question: {qa['question']}\nOptions:\n0: {qa['choices'][0]}\n1: {qa['choices'][1]}\n2: {qa['choices'][2]}\n3: {qa['choices'][3]}\n"""  # Answer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not answer in the form a sentence; answer only with a single number as your output."""

                    print(f"Processing question {idx+1}/{len(batch)}...")
                    # print(question)

                    # try:
                    is_correct = self.process_question(question, qa["answer"], num_result_files)

                    if is_correct:
                        batch_num_correct += 1
                        all_num_correct += 1
                    batch_total += 1
                    all_total += 1
                    # except:
                    #     continue

                # Calculate accuracy
                batch_accuracy = (batch_num_correct / batch_total) * 100
                print(f"Batch {batch_i}, Accuracy: {batch_accuracy:.2f}%")
                batch_accuracies.append(batch_accuracy)
                
                running_accuracy = (all_num_correct / all_total) * 100
                running_accuracies.append(running_accuracy)
                
                num_traj_files = len([f for f in os.listdir("trajectories/normal_accuracy_trajectories/")])
                traj_filename = f"trajectories/normal_accuracy_trajectories/traj{num_traj_files}"
                
                with open(traj_filename,  "a") as traj_file:
                    traj_file.write(f"{batch_accuracy} {running_accuracy}")
            
            plt.figure(figsize=(8, 5))
            plt.plot(batch_accuracies, marker='o')
            plt.xlabel('Batch number')
            plt.ylabel('Accuracy (%)')
            plt.title('Evaluation Accuracy, Subjects = {subjects}')
            plt.grid(True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_plot_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")

            plt.show()

        return batch_accuracies

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
    # system.evaluate_gpt4_on_mmlu_mixed(sample_size=4)
    system.full_eval_mmlu_mixed(batch_size = 20, max_iters = 5)

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
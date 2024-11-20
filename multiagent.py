from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager

import time

# external API for updating reputation scores
def update_reputation(question, agent_correctness):
    pass

def main():
    # Initialize agents
    llm_config = {"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    num_agents = 4
    agents = []

    for i in range(num_agents):
        agent = ConversableAgent(
            name=f"Agent {i}",
            system_message='''You are a collaborative agent that will engage in constructive discussions to come to a collective answer.
                            You will be given information about how other agents performed on each question, 
                            which you should use to decide how much to trust their information.''',
            llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
            # human_input_mode="NEVER",
            silent = False
        )

        agents.append(agent)
    
    group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=6,
            speaker_selection_method = "round_robin"
        )
    print(f"{type(group_chat)=}")
    chat_manager = GroupChatManager(group_chat)

    questions = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"}
    ]

    for qa in questions:
        question = qa["question"]
        correct_answer = qa["answer"]
        print(f"\nQuestion: {question}")
        chat_manager.run_chat(messages = [question])

        # print(f"\nQuestion: {question}")

        # # Each agent gives their final answer
        # final_answers = {}
        # for name, agent in agents.items():
        #     reputation_context = get_reputation_context(name, reputation_scores)
        #     agent_message = f"{reputation_context}\n\nPlease provide your final answer to the question."
        #     response = agent.step(agent_message)
        #     final_answers[name] = response.content.strip()
        #     print(f"{name}'s final answer: {final_answers[name]}")
        #     time.sleep(1)

        # # Agents vote on the final answer (collectively determine the most common final answer)
        # vote_values = list(final_answers.values())
        # final_answer = max(set(vote_values), key=vote_values.count)
        # print(f"\nFinal answer after voting: {final_answer}")

        # # Check if each agent's final answer was correct
        # for name in agent_names:
        #     is_correct = (final_answers[name].lower() == correct_answer.lower())
        #     reputation = update_reputation(name, question, is_correct, reputation_scores)
        #     print(f"{name}'s final answer was {'correct' if is_correct else 'incorrect'}. Updated reputation: {reputation}")

        # # Pass the correct answer to agents (for future context)
        # for name, agent in agents.items():
        #     agent.receive_message(
        #         sender='System',
        #         message=f"The correct answer was: {correct_answer}",
        #         silent=True
        #     )

    # print("\nFinal reputation scores:")
    # for name, score in reputation_scores.items():
    #     print(f"{name}: {score}")

if __name__ == "__main__":
    import os
    main()

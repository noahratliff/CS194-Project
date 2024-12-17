from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
import os

import time


# external API for updating reputation scores
def update_reputation(question, agent_correctness):
    pass


def get_reputation_context(agent_name, reputation_scores):
    pass


def main(use_reputation=False):
    # Initialize agents
    llm_config = {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}
        ]
    }
    num_agents = 4
    agents = [
        ConversableAgent(
            name=f"Facilitator",
            llm_config=llm_config,
            system_message="""You are a facilitator that will guide the conversation and provide information to the agents. When it is your turn, you will ask the agents for the final answer. You will not provide your own answers to the questions, but simply ask "What is your final answer? Give just the answer please." """,
            silent=False,
        )
    ]

    for i in range(num_agents):
        agent = ConversableAgent(
            name=f"Agent{i}",
            system_message="""You are a collaborative agent that will engage in constructive discussions to come to a collective answer.
                            You will be given information about how other agents performed on each question, 
                            which you should use to decide how much to trust their information.""",
            llm_config={
                "config_list": [
                    {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}
                ]
            },
            # human_input_mode="NEVER",
            silent=False,
        )

        agents.append(agent)

    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=(num_agents + 1) * 2,
        speaker_selection_method="round_robin",
    )
    print(f"{type(group_chat)=}")
    chat_manager = GroupChatManager(group_chat)

    questions = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"},
    ]

    reputation_scores = []

    for qa in questions:
        question = qa["question"]
        correct_answer = qa["answer"].lower()
        print(f"\nQuestion: {question}")
        chat_manager.run_chat(
            messages=[{"role": "user", "content": question}],
            config=group_chat,
            sender=agents[0],
        )

        # Get the final answer from the agents

        final_answers = {}
        for message in group_chat.messages[-num_agents:]:
            name = message["name"]
            final_answers[name] = message["content"].strip().lower()
            print(f"{name}'s final answer: {final_answers[name]}")

        vote_values = list(final_answers.values())
        final_answer = max(set(vote_values), key=vote_values.count)

        is_correct = final_answer == correct_answer
        print(
            f"The group's final answer was {'correct' if is_correct else 'incorrect'}"
        )

        for agent in agents:
            if agent.name == "Facilitator":
                continue
            is_correct = final_answers[agent.name] == correct_answer
            print(
                f"{agent.name}'s final answer was {'correct' if is_correct else 'incorrect'}"
            )

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

        break

    # print("\nFinal reputation scores:")
    # for name, score in reputation_scores.items():
    #     print(f"{name}: {score}")


if __name__ == "__main__":

    main()

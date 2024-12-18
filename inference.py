from datasets import load_dataset
import evaluate
import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID"),
)

def load_mmlu(subject: str):
    """
    Load the MMLU dataset for a specific subject.
    """
    dataset = load_dataset("cais/mmlu", subject) # , subject, trust_remote_code=True)
    # Combine validation and test sets for evaluation
    questions = dataset["validation"] + dataset["test"]
    formatted_questions = [
        {
            "question": q["question"],
            "options": [f"A. {q['answerA']}", f"B. {q['answerB']}", f"C. {q['answerC']}", f"D. {q['answerD']}"],
            "correct": q["correct"],
        }
        for q in questions
    ]
    return formatted_questions

def evaluate_question(question) -> str:
    """
    Evaluate a single multiple-choice question using GPT-3.5.
    """
    prompt = (
        f"Question: {question['question']}\n"
        f"Options:\n"
        f"{question['options'][0]}\n"
        f"{question['options'][1]}\n"
        f"{question['options'][2]}\n"
        f"{question['options'][3]}\n"
        "Answer with the letter corresponding to the correct choice (e.g., A, B, C, or D)."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at answering multiple-choice questions."},
                {"role": "user", "content": prompt}]
            )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error querying GPT-3.5: {e}")
        return None

def evaluate_dataset(dataset):
    """
    Evaluate the MMLU dataset and calculate accuracy.
    """
    total = len(dataset)
    correct = 0

    for idx, question in enumerate(dataset):
        print(f"Evaluating question {idx + 1}/{total}...")
        model_answer = evaluate_question(question)

        if model_answer == question["correct"]:
            correct += 1
        else:
            print(f"Wrong answer. Expected {question['correct']}, but got {model_answer}.")

    accuracy = correct / total * 100
    return {"total": total, "correct": correct, "accuracy": accuracy}

if __name__ == "__main__":
    # Choose an MMLU subject to evaluate
    subject = "college_mathematics"  # Replace with your desired subject (e.g., "biology", "world_religions")
    
    print(f"Loading MMLU subject: {subject}...")
    dataset = load_mmlu(subject)
    
    print(f"Evaluating GPT-3.5 on {subject}...")
    results = evaluate_dataset(dataset)

    print(f"Results for {subject}:")
    print(f"Total Questions: {results['total']}")
    print(f"Correct Answers: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
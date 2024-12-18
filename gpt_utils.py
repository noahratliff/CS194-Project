import openai
from openai import OpenAI
import random
import os
from datasets import load_dataset
from dotenv import load_dotenv

# Set your OpenAI API Key
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID"),
)
# Function to query GPT-4 API
def query_gpt4(prompt, model="gpt-4"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at answering multiple-choice questions."},
                {"role": "user", "content": prompt}
            ],            
            temperature=0.0,  # Deterministic output for accuracy testing
        )
        print("RESPONSE", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying GPT-4: {e}")
        return None

# Function to evaluate GPT-4 on a dataset subset
def evaluate_gpt4_on_mmlu(subject="high_school_mathematics", sample_size=20):
    # Load the MMLU dataset
    dataset = load_dataset("cais/mmlu", subject, split="test")

    # Sample a subset of questions
    sampled_questions = random.sample(list(dataset), sample_size)

    correct_count = 0
    for idx, question in enumerate(sampled_questions):
        prompt = f"""Question: {question['question']}\nOptions:\n0: {question['choices'][0]}\n1: {question['choices'][1]}\n2: {question['choices'][2]}\n3: {question['choices'][3]}\nAnswer with the number corresponding to the correct choice (0, 1, 2, or 3). Do not generate any additional text."""
        
        print(f"Processing question {idx+1}/{sample_size}...")
        # print(question)
        gpt4_answer = query_gpt4(prompt)

        # Compare GPT-4's answer to the ground truth
        print(gpt4_answer)
        print(question["answer"])

        if gpt4_answer == str(question['answer']):
            correct_count += 1
        else:
            print(f"Wrong Answer: {gpt4_answer} (Correct: {question['answer']})")
    
    # Calculate accuracy
    accuracy = (correct_count / sample_size) * 100
    print(f"\nAccuracy on {subject}: {accuracy:.2f}%")
    return accuracy

# Main execution
if __name__ == "__main__":
    # Example: Evaluate on High School Mathematics (subset of 20 samples)
    subject = "high_school_mathematics"
    sample_size = 20
    evaluate_gpt4_on_mmlu(subject=subject, sample_size=sample_size)

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def calculate_cumulative_accuracy(df, answer_column_prefix):
    # Calculate cumulative accuracy for each question
    correct_answers = df['Correct Answer']
    cumulative_accuracies = []
    correct_count = 0
    
    # Debug: Print the column prefix we're looking for
    print(f"\nCalculating accuracy for prefix: {answer_column_prefix}")
    matching_columns = df.filter(like=answer_column_prefix).columns
    print(f"Matching columns found: {matching_columns}")
    
    for index, row in df.iterrows():
        answers = row.filter(like=answer_column_prefix)
        print(f"\nQuestion {index + 1}:")
        print(f"Correct answer: {correct_answers[index]}")
        print(f"Agent answers: {answers.values}")
        
        # Debug: Print each comparison
        correct_in_this_round = sum(answers == correct_answers[index])
        print(f"Correct answers this round: {correct_in_this_round}")
        
        correct_count += correct_in_this_round
        cumulative_accuracy = correct_count / ((index + 1) * len(answers))
        print(f"Cumulative accuracy: {cumulative_accuracy}")
        
        cumulative_accuracies.append(cumulative_accuracy)
    
    return cumulative_accuracies

def plot_cumulative_accuracies(initial_cumulative_accuracies, final_cumulative_accuracies_no_reputation, final_cumulative_accuracies_with_reputation):
    # Debug: Print lengths and values of accuracy arrays
    print("\nPlotting accuracies:")
    print(f"Initial accuracies ({len(initial_cumulative_accuracies)}): {initial_cumulative_accuracies}")
    print(f"No reputation accuracies ({len(final_cumulative_accuracies_no_reputation)}): {final_cumulative_accuracies_no_reputation}")
    print(f"With reputation accuracies ({len(final_cumulative_accuracies_with_reputation)}): {final_cumulative_accuracies_with_reputation}")
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 6))

    # Use a modern color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, and green

    # Plot cumulative accuracies
    plt.plot(initial_cumulative_accuracies, label='Initial Cumulative Accuracy', marker='o', color=colors[0], linewidth=2)
    plt.plot(final_cumulative_accuracies_no_reputation, label='Final Cumulative Accuracy (No Reputation)', marker='x', color=colors[1], linewidth=2)
    plt.plot(final_cumulative_accuracies_with_reputation, label='Final Cumulative Accuracy (With Reputation)', marker='s', color=colors[2], linewidth=2)

    # Customize the plot
    plt.title('Cumulative Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Question Number', fontsize=14)
    plt.ylabel('Cumulative Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Despine the plot
    sns.despine()

    # Save the plot
    plt.tight_layout()
    plt.savefig('results/cumulative_accuracy_over_time.png', dpi=300)
    plt.show()

def plot_results(csv_file):
    # Load the results from the CSV file
    df = pd.read_csv(csv_file)
    
    # Debug: Print all column names and first few rows
    print("\nDataFrame Info:")
    print(df.info())
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # Calculate cumulative accuracies
    initial_cumulative_accuracies = calculate_cumulative_accuracy(df, 'Initial Answer')
    final_cumulative_accuracies_no_reputation = calculate_cumulative_accuracy(df, 'Final Answer No Reputation')
    final_cumulative_accuracies_with_reputation = calculate_cumulative_accuracy(df, 'Final Answer With Reputation')

    # Plot cumulative accuracies
    plot_cumulative_accuracies(initial_cumulative_accuracies, 
                             final_cumulative_accuracies_no_reputation, 
                             final_cumulative_accuracies_with_reputation)

if __name__ == "__main__":
    # Ensure the results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Directory '{results_dir}' does not exist. Please ensure the results are saved correctly.")
    else:
        csv_file = os.path.join(results_dir, "results.csv")
        if os.path.exists(csv_file):
            plot_results(csv_file)
        else:
            print(f"CSV file '{csv_file}' not found. Please ensure the results are saved correctly.") 
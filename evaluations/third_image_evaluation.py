from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import json

def tokenize(text):
    return text.lower().strip().split()

def compute_metrics(ground_truth, prediction):
    gt_tokens = tokenize(ground_truth)
    pred_tokens = tokenize(prediction)

    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)

    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Words in Ground Truth": len(gt_tokens),
        "Words in Prediction": len(pred_tokens),
        "Correct Words": true_positives,
    }

def evaluate_multiple_llms(ground_truth, llm_predictions):
    """
    ground_truth: string of the true text
    llm_predictions: dict with keys as LLM names and values as predicted strings
    Returns: dict of results per LLM
    """
    results_per_llm = {}

    for llm_name, prediction in llm_predictions.items():
        metrics = compute_metrics(ground_truth, prediction)
        results_per_llm[llm_name] = metrics

    return results_per_llm

ground_truth = """
Mini Course 3: Basics of Probability Theory


Probability theory is the branch of mathematics concerned with the analysis of random phenomena. The outcome of a random experiment is described by a probability distribution.

P(A∪B)=P(A)+P(B)−P(A∩B)

This formula is essential in determining the probability of either event A or event B occurring. It avoids double-counting the overlap between the two events, represented by P(A∩B)
"""

llm_predictions = {
    "ChatGPT": """
    Mini course 3: Basics of probability theory

Probability theory is the branch of mathematics concerned with the analysis of random phenomena.

The outcome of a random experiment is described by a probability distribution.

𝑃
(
𝐴
∪
𝐵
)
=
𝑃
(
𝐴
)
+
𝑃
(
𝐵
)
−
𝑃
(
𝐴
∩
𝐵
)
P(A∪B)=P(A)+P(B)−P(A∩B)
This formula is essential in determining the probability of either event A or event B occurring; it avoids double counting the overlap between the two events, represented by 
𝑃
(
𝐴
∩
𝐵
)
P(A∩B).
    """,
    "Claude": """
Mini Course 3: Basics of Probability Theory
Probability theory is the branch of mathematics concerned with the analysis of random phenomena.
The outcome of a random experiment is described by a probability distribution.
Addition Rule for Probability
Formula:
P(A∪B) = P(A) + P(B) - P(A∩B)
Explanation:
This formula is essential in determining the probability of either event A or event B occurring. It avoids double counting the overlap between the two events, represented by P(A∩B).
    """,
    "Gemini": """

Mini course 3: Basics of probability theory

Probability theory is the branch of mathematics concerned with the analysis of random phenomena.
The outcome of a random experiment is described by a probability distribution.

P(A∪B)=P(A)+P(B)−P(A∩B)

This formula is essential in determining the probability of either event A or event B occurring. It avoids double counting the overlap between the two events, represented by P(A∩B).
    """,
    "Deepseek": """
    Mini Course 3: Basics of Probability Theory

Probability theory is the branch of mathematics concerned with the analysis of random phenomena. The outcome of a random experiment is described by a probability distribution.

Key Formula
P
(
A
∪
B
)
=
P
(
A
)
+
P
(
B
)
−
P
(
A
∩
B
)
P(A∪B)=P(A)+P(B)−P(A∩B)
This formula is essential in determining the probability of either event 
A
A or event 
B
B occurring. It avoids double-counting the overlap between the two events, represented by 
P
(
A
∩
B
)
P(A∩B).
    """,
    "Grok": """
Mini Course 3: Basics of Probability Theory
Probability theory is the branch of mathematics concerned with the analysis of random phenomena. The outcome of a random experiment is described by a probability distribution.
$ P(A \cup B) = P(A) + P(B) - P(A \cap B) $
This formula is essential in determining the probability of either event A or event B occurring. It avoids double counting the overlap between the two events, represented by $ P(A \cap B) $.
    """,
}

results = evaluate_multiple_llms(ground_truth, llm_predictions)

for llm, metrics in results.items():
    print(f"Results for {llm}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

# Optionally, save results to JSON
with open("llm_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

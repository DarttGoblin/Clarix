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
These metrics allow analysis to gain quick insights into the distribution and variability of the data, guiding further analysis or model selection.

Mini Course 4: Matrix Multiplication

Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformation, 3D graphics, and machine learning models.

Matrix A (2x3):
[1 2 3]
[4 5 6]

Matrix B (3x2):
[7 8]
[9 10]
[11 12]

Matrix C = A x B (2x2):
[58 64]
[139 154]
"""

llm_predictions = {
    "ChatGPT": """
Mini Course 4: Matrix Multiplication

Paragraph
Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformation, 3D graphics, and machine learning models.

Matrix A (2×3)	Matrix B (3×2)	Matrix C = B × A (2×2)
[ [1 \ 2 \ 3]	[ [7 \ 8]	[ [58 \ 64]
[4 \ 5 \ 6] ]	[9 \ 10]	[139 \ 154] ]
[11 \ 12] ]
    """,
    "Claude": """
These metrics allow analysis to gain quick insights into the distribution and variability of the data, guiding further analysis or model selection.
Mini Course 4: Matrix Multiplication
Paragraph
Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformation, 3D graphics and machine learning models.
Matrix Multiplication Example
Matrix A (2×3)Matrix B (3×2)Matrix C = B × A (2×2)[1  2  3][7   8][58   64][4  5  6][9  10][139  154][11 12]
    """,
    "Gemini": """

These metrics allow analysis to gain quick insights into the distribution and variability of the data, guiding further analysis or model selection.

Mini Course 4: Matrix Multiplication

Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformation, 3D graphics, and machine learning models.

Matrix A (2x3)	Matrix B (3x2)	Matrix C = B x A (2x2)
[ 
1
4
​
  
2
5
​
  
3
6
​
 ]	 

​
  
7
9
11
​
  
8
10
12
​
  

​
 	[ 
58
139
​
  
64
154
​
 ]
    """,
    "Deepseek": """
Mini Course 4: Matrix multiplication

These metrics allow analysis to gain quick insights into the distribution and variability of the data, guiding further analysis or model selection.

Paragraph
Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformations, 3d graphics and machine Learning models.

Matrix A (2x3)

[
1
2
3
4
5
6
]
[ 
1
4
​
  
2
5
​
  
3
6
​
 ]
Matrix B (3x2)

[
7
8
9
10
11
12
]
​
  
7
9
11
​
  
8
10
12
​
  
​
 
Matrix C = B x A (2x2)

[
5
8
64
139
154
]
[ 
5
139
​
  
8
154
​
  
64
 ]
    """,
    "Grok": """
Mini Course 4: Matrix Multiplication

These metrics allow analysis to gain quick insights into the distribution and variability of the data, guiding further analysis or model selection.

Paragraph

Matrix multiplication is a core operation in linear algebra. It is used extensively in data transformation, 3D graphics, and machine learning models.


Matrix A (2x3)	Matrix B (3x2)	Matrix C = B × A (2x2)
[ 1  2  3 ]	[ 7  8 ]	[ 58  64 ]
[ 4  5  6 ]	[ 9  10 ]	[ 139  154 ]
[ 11 12 ]
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

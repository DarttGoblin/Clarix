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
Mini Course 6: Newton's Laws of Motion (Py Docs)

Newton's Laws of Motion describe the relationship between a body and the forces acting on it, and its motion in response to those forces. These laws are fundamental in classical mechanics.

Equation:

Newton's Second Law:
F = ma

Where:

F is the force applied (in Newtons)  
m is the mass of the object (in Kg)  
a is the acceleration produced (in m/s²)  

This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science, and mechanics.
"""

llm_predictions = {
    "ChatGPT": """
Mini Course 6: Newton’s Laws of Motion (Physics)

Newton’s Laws of Motion describe the relationship between a body and the forces acting on it, and its motion in response to those forces. These laws are fundamental in classical mechanics.

Equation:

Newton’s Second Law

f = ma

Where:

f is the force applied (in Newtons)

m is the mass of the object (in kg)

a is the acceleration produced (in m/s²)

This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science, and mechanics.
    """,
    "Claude": """
Mini Course 6: Newton's Laws of Motion (Physics)
Newton's laws of motion describe the relationship between a body and the forces acting on it, and its motion in response to those forces. These laws are fundamental in classical mechanics.
Equation:
Newton's Second Law:
F = ma
Where:

F is the force applied (in Newtons)
m is the mass of the object (in kg)
a is the acceleration produced (in m/s²)

Explanation:
This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science and mechanics.
    """,
    "Gemini": """

Mini course 6: Newton's Laws of Motion (Physics)

Newton's Laws of motion describe the relationship between a body and the forces acting on it, and its motion in response to those forces. These laws are fundamental in classical mechanics.

Equation:
Newton's Second Law:
F=ma
Where:

F is the force applied (in Newtons)
m is the mass of the object (in Kg)
a is the acceleration produced (in m/s 
2
 )
This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science, and mechanics.
    """,
    "Deepseek": """
Ninicourse 6: Newton's laws of fiction (by Brics)

Newton Laws of motion describe the relationship between a body and the forces acting on it / and its motion in response to those forces. These laws are fundamental in classical mechanics.

Equation:
Newton's second law:

Where:

f
=
m
a
f=ma
f
f is the force applied (in Newtons)

m
m is the mass of the object (in kg)

a
a is the acceleration produced (in 
m
/
s
2
m/s 
2
 )

This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science and mechanics.
    """,
    "Grok": """
Mini Course 6: Newton's Laws of Motion (Physics)
Newton's Laws of motion describe the relationship between a body and the forces acting on it, and its motion in response to those forces. These laws are fundamental in classical mechanics.
Equation:
Newton's Second Law:
$ F = m a $
Where:

$ F $ is the force applied (in Newtons)
$ m $ is the mass of the object (in kg)
$ a $ is the acceleration produced (in m/s²)

This equation implies that the acceleration of an object is directly proportional to the net force and inversely proportional to its mass. It is used in engineering, space science, and mechanics.
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

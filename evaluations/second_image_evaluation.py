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
Mini Course 2: Descriptive Statistics Basics

Descriptive statistics summarize and organize characteristics of  dataset. They include measures of central tendency, dispersion, and shape of the distribution.

Measure	 Description	Formula
Mean	 Average value	xˉ=1n∑xixˉ=n1​∑xi​
Median	 Middle value	N/A (position-based)
Standard Deviation	Spread of data around the mean	1n∑(xi−xˉ)2n1​∑(xi​−xˉ)2
​
Variance	Average squared deviation	1n∑(xi−xˉ)2n1​∑(xi​−xˉ)2
"""

llm_predictions = {
    "ChatGPT": """
Mini Course 2: Descriptive Statistics Basics

Descriptive statistics summarise and organise characteristics of datasets. They include measures of tendency, dispersion, and shape of the distribution.

Measure	Description	Formula
Mean	Average value	
𝑥
ˉ
=
1
𝑛
∑
𝑥
𝑖
x
ˉ
 = 
n
1
​
 ∑x 
i
​
 
Median	Middle value	N/A (position-based)
Standard deviation	Spread of data around the mean	
1
𝑛
∑
(
𝑥
𝑖
−
𝑥
ˉ
)
2
n
1
​
 ∑(x 
i
​
 − 
x
ˉ
 ) 
2
 
​
 
Variance	Average squared deviation	
1
𝑛
∑
(
𝑥
𝑖
−
𝑥
ˉ
)
2
n
1
​
 ∑(x 
i
​
 − 
x
ˉ
 ) 
2
    """,
    "Claude": """
    Mini Course 2: Descriptive Statistics Basics
Descriptive statistics summarise and organise characteristics of datasets. They include measures of tendency, dispersion, and shape of the distribution.
Statistical Measures Table
MeasureDescriptionFormulaMeanAverage valuex̄ = (1/n) Σ xᵢMedianMiddle valueN/A position-basedStandard deviationSpread of data around the mean√[(1/n) Σ(xᵢ - x̄)²]VarianceAverage squared deviation(1/n) Σ(xᵢ - x̄)²
    """,
    "Gemini": """

Mini Course 2: Descriptive Statistics basics

Descriptive statistics summarize and organize characteristics of datasets. They include measures of tendency, dispersion, and shape of the distribution.
Measure	Description	Formula
Mean	Average value	 
x
ˉ
 = 
n
1
​
 ∑x 
i
​
 
Median	Middle value	N/A position-based
Standard deviation	Spread of data around the mean	 
n
1
​
 ∑(x 
i
​
 − 
x
ˉ
 ) 
2
 

​
 
Variance	Average squared deviation	 
n
1
​
 ∑(x 
i
​
 − 
x
ˉ
 ) 
2
    """,
    "Deepseek": """
Mini Course 2: Descriptive Statistics Basics

Descriptive statistics summarize and organize characteristics of datasets. They include measures of central tendency, dispersion, and the shape of the distribution.

Key Measures
Measure	Description	Formula
Mean	Average value	
x
‾
=
1
n
∑
x
i
x
 = 
n
1
​
 ∑x 
i
​
 
Median	Middle value	N/A (position-based)
Standard Deviation	Spread of data around the mean	
1
n
∑
(
x
i
−
x
‾
)
2
n
1
​
 ∑(x 
i
​
 − 
x
 ) 
2
 
​
 
Variance	Average squared deviation	
1
n
∑
(
x
i
−
x
‾
)
2
n
1
​
 ∑(x 
i
​
 − 
x
 ) 
2
    """,
    "Grok": """
Mini Course 2: Descriptive Statistics Basics

Descriptive statistics summarise and organise characteristics of datasets. They include measures of tendency, dispersion, and shape of the distribution.































MeasureDescriptionFormulaMeanAverage value$\bar{x} = \frac{1}{n} \sum x_i$MedianMiddle valueN/A (position-based)Standard deviationSpread of data around the mean$\sqrt{\frac{1}{n} \sum (x_i - \bar{x})^2}$VarianceAverage squared deviation$\frac{1}{n} \sum (x_i - \bar{x})^2$
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

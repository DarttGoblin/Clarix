import matplotlib.pyplot as plt

# All models and their F1 scores
models = ['Gemini', 'ChatGPT', 'Deepseek', 'Grok', 'Claude', 'PaddleOCR', 'EasyOCR', 'TesseractOCR']
f1_scores = [0.8466, 0.8162, 0.7965, 0.7645, 0.7550, 0.3333, 0.1158, 0.0000]

# Plot setup
plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color='skyblue')

# Annotate bars with values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)

# Axis and title
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison of LLMs and OCR Models")
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save as image
plt.tight_layout()
plt.savefig("f1_scores_all_models.png", dpi=300)
plt.show()
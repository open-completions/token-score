def compute_parity(models, languages):
    parity_scores = {}

    for model_name, results in models.items():
        # Compute the average compression value for each language
        averages = {lang: results[lang]["Compression"] for lang in languages}

        # Calculate pairwise parities and average them
        parity_sum = 0
        count = 0
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                parity_sum += abs(averages[languages[i]] - averages[languages[j]])
                count += 1

        # Store the average parity for the model
        parity_scores[model_name] = parity_sum / count

    return parity_scores


# Assuming 'models' is a dictionary containing each model's metrics per language
models = {
    "GPT-4": {
        "Python": {"Compression": 4.19},
        "JavaScript": {"Compression": 3.43},
        "Go": {"Compression": 3.44},
        "C++": {"Compression": 3.87},
        "Java": {"Compression": 4.56},
    },
    "Codex": {
        "Python": {"Compression": 3.16},
        "JavaScript": {"Compression": 2.75},
        "Go": {"Compression": 2.62},
        "C++": {"Compression": 2.95},
        "Java": {"Compression": 3.37},
    },
    "Stable Code": {
        "Python": {"Compression": 3.3},
        "JavaScript": {"Compression": 2.92},
        "Go": {"Compression": 2.85},
        "C++": {"Compression": 3.12},
        "Java": {"Compression": 3.61},
    },
    "Replit Code": {
        "Python": {"Compression": 3.62},
        "JavaScript": {"Compression": 3.16},
        "Go": {"Compression": 2.94},
        "C++": {"Compression": 3.38},
        "Java": {"Compression": 4.01},
    },
    "Code LLaMa": {
        "Python": {"Compression": 3.07},
        "JavaScript": {"Compression": 2.58},
        "Go": {"Compression": 2.51},
        "C++": {"Compression": 2.84},
        "Java": {"Compression": 3.34},
    },
}

languages = ["Python", "JavaScript", "Go", "C++", "Java"]
parity_scores = compute_parity(models, languages)

# Output the parity score for each model
for model, score in parity_scores.items():
    print(f"Parity score for {model}: {score}")

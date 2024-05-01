import matplotlib.pyplot as plt

# Data
epitopes = ["IVTDFSVIK", "KLGGALQAK", "GILGFVFTL", "ELAGIGILTV", "HPVTKYIM",
            "RAQAPPPSW", "RLRAEAQVK", "AVFDRKSDAK", "RAKFKQLL", "TPRVTGGGAM"]
one_hot = [0.91, 0.88, 0.96, 0.91, 0.85, 0.98, 0.83, 0.84, 0.94, 0.93]
giana = [0.91, 0.91, 0.97, 0.89, 0.86, 0.98, 0.85, 0.83, 0.94, 0.96]
bert = [0.91, 0.92, 0.96, 0.89, 0.88, 0.98, 0.84, 0.84, 0.95, 0.92]
bert_mlm = [0.91, 0.93, 0.97, 0.90, 0.88, 0.98, 0.86, 0.85, 0.95, 0.95]

# Bar width
bar_width = 0.2

# X positions
r1 = range(len(epitopes))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # hex colors from the image

# Plotting the data with the new color scheme
fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(r1, one_hot, color=colors[0 % len(colors)], width=bar_width, label='One-Hot')
ax.bar(r2, giana, color=colors[1 % len(colors)], width=bar_width, label='Giana')
ax.bar(r3, bert, color=colors[2 % len(colors)], width=bar_width, label='Bert')
ax.bar(r4, bert_mlm, color=colors[3 % len(colors)], width=bar_width, label='Bert-MLM')  # silver for contrast

# Adding labels and title
ax.set_xlabel('Epitope', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy of Prediction Models by Epitope', fontsize=16)
ax.set_xticks([x + 1.5 * bar_width for x in range(len(epitopes))])
ax.set_xticklabels(epitopes, rotation=45)
ax.legend()

plt.show()

# New Data for F1 Scores
f1_scores = {
    'One-Hot': [0.7900, 0.6800, 0.9300, 0.8200, 0.7200, 0.9600, 0.5100, 0.5300, 0.8700, 0.8700],
    'Giana': [0.6500, 0.6400, 0.8800, 0.5500, 0.5000, 0.9300, 0.1800, 0.1300, 0.7700, 0.8600],
    'Bert': [0.7954, 0.8258, 0.9268, 0.7411, 0.7272, 0.9617, 0.5111, 0.5069, 0.8864, 0.8512],
    'Bert-MLM': [0.7938, 0.8446, 0.9365, 0.7730, 0.7272, 0.9617, 0.5504, 0.5448, 0.8972, 0.8935]
}

# Plotting the F1 Scores with the adjusted color scheme
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each model's F1 scores
for i, (model, scores) in enumerate(f1_scores.items()):
    ax.bar([x + (i * bar_width) for x in r1], scores, color=colors[i % len(colors)], width=bar_width, label=model)

# Adding labels and title
ax.set_xlabel('Epitope', fontsize=14)
ax.set_ylabel('F1 Score', fontsize=14)
ax.set_title('F1 Score of Prediction Models by Epitope', fontsize=16)
ax.set_xticks([x + 1.5 * bar_width for x in range(len(epitopes))])
ax.set_xticklabels(epitopes, rotation=45)
ax.legend()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import re

# Function to strip XML tags from text
def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# Load tasks
with open('data/raw/mslatte_tasks.txt', 'r', encoding='utf-8', errors='replace') as f:
    tasks = [line.strip() for line in f if line.strip()]

# Load questions
with open('data/raw/msmarco_questions_train_uniq.txt', 'r', encoding='utf-8', errors='replace') as f:
    questions = [line.strip() for line in f if line.strip()]

# Load deadlines from timeml_deadline_sentences.txt
deadlines = []
with open('data/raw/timeml_deadline_sentences.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            text = '\t'.join(parts[1:])  # In case there are tabs in text
            plain_text = strip_tags(text)
            deadlines.append(plain_text)

# Create DataFrame
data = []
for task in tasks:
    data.append({'text': task, 'label': 'Task'})

for question in questions:
    data.append({'text': question, 'label': 'Question'})

for deadline in deadlines:
    data.append({'text': deadline, 'label': 'Deadline'})

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('notes_data.csv', index=False)

# EDA
print("Missing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['label'].value_counts())

# Plot class distribution
df['label'].value_counts().plot(kind='bar', title='Count of Notes per Category')
plt.savefig('class_distribution.png')
plt.close()

# Text length analysis
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

print("\nAverage word count per category:")
print(df.groupby('label')['word_count'].mean())

# Plot histograms for word counts
for label in df['label'].unique():
    subset = df[df['label'] == label]
    plt.figure()
    subset['word_count'].hist(bins=20)
    plt.title(f'Word Count Distribution for {label}')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig(f'word_count_{label}.png')
    plt.close()
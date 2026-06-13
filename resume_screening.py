import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv(r"C:\Users\m yadaiah\Desktop\Resume screeing\Resume.csv")

# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

df["Resume"] = df["Resume_str"].apply(clean_text)

# Job Description
job_desc = """
Data Scientist
Skills: Python, SQL, Machine Learning, Pandas, Statistics
"""

job_desc = clean_text(job_desc)

# TF-IDF Similarity
documents = [job_desc] + df["Resume"].tolist()

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

df["Match Score"] = scores * 100

# Skill Extraction
skills = ["python", "sql", "machine learning", "pandas", "statistics"]

def extract_skills(text):
    return [skill for skill in skills if skill in text]

df["Skills Found"] = df["Resume"].apply(extract_skills)

# Skill Gap
required = extract_skills(job_desc)

df["Missing Skills"] = df["Skills Found"].apply(
    lambda x: [s for s in required if s not in x]
)

# Ranking
result = df.sort_values("Match Score", ascending=False)
import matplotlib.pyplot as plt

top = result.head(5)

plt.figure(figsize=(8,5))
plt.bar(top["Category"], top["Match Score"])
plt.xlabel("Category")
plt.ylabel("Match Score (%)")
plt.title("Top 5 Candidate Match Scores")

plt.xticks(rotation=45)

# SAVE AS PNG FILE
plt.savefig("candidate_ranking.png")

plt.show()

# Top 5 Candidates
print("\nTop 5 Candidates\n")

for i, row in result.head(5).iterrows():

    print("Category:", row["Category"])
    print("Match Score:", round(row["Match Score"], 2), "%")
    print("Skills Found:", row["Skills Found"])
    print("Missing Skills:", row["Missing Skills"])
    print("-" * 50)

# Save Results
result.to_csv("candidate_ranking.csv", index=False)

print("\nResults saved as candidate_ranking.csv")
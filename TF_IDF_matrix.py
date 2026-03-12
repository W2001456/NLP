import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



def read_simple(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


docs = []
names = ['1.txt', '2.txt', '3.txt']
txt4 = 'I need to learn more about science, philosophy, geography, and related fields because I enjoy these.'
for name in names:
    try:
        content = read_simple(f'{name}')
        print(type(content))
        docs.append(content)
        print(f"Successfully read document{name}: {len(content)}")
    except:
        print(f"no {name}")
        docs.append("")
docs.append(txt4)
names.append('4.txt')
# F-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
tfidf_matrix = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

# DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=names
)

print("\nTF-IDF matrix")
print(tfidf_df)

"""
result
TF-IDF matrix
       development  geography    global  ...  political    rural   urban
1.txt       0.3711   0.536396  0.039478  ...   0.000000  0.30925  0.3711
2.txt       0.0000   0.000000  0.126377  ...   0.989968  0.00000  0.0000
3.txt       0.0000   0.000000  0.237542  ...   0.000000  0.00000  0.0000
4.txt       0.0000   1.000000  0.000000  ...   0.000000  0.00000  0.0000

[4 rows x 10 columns]

"""

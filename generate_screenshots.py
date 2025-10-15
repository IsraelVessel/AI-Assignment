"""Generate screenshots for Iris report, spaCy NER output, and MNIST sample images.
Run with the venv python: python generate_screenshots.py
"""
import os
import random
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

SCREEN_DIR = 'screenshots'
os.makedirs(SCREEN_DIR, exist_ok=True)

# Iris report image
print('Generating Iris screenshot...')
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Save text report to PNG
fig, ax = plt.subplots(figsize=(6,4))
ax.axis('off')
ax.text(0, 1, 'Iris Classification Report', fontsize=14, weight='bold')
ax.text(0, 0.9, report, fontsize=10, family='monospace')
plt.tight_layout()
plt.savefig(os.path.join(SCREEN_DIR, 'iris_report.png'), dpi=150)
plt.close()

# spaCy NER output image
print('Generating spaCy screenshot...')
try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception as e:
    print('spaCy not available:', e)
    # write a placeholder image
    fig, ax = plt.subplots(figsize=(6,2))
    ax.axis('off')
    ax.text(0,0.5,'spaCy not installed in this environment. Run: pip install spacy && python -m spacy download en_core_web_sm', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREEN_DIR, 'spacy_ner.png'), dpi=150)
    plt.close()
else:
    nlp = spacy.load('en_core_web_sm')
    reviews = [
        'I love the Acme SmartWatch, the battery lasts all week and the strap is comfortable.',
        'The Zeta Vacuum is noisy and stopped working after two weeks. Terrible experience.',
        'Great headphones by SoundMax — amazing bass and clear mids.'
    ]
    out_lines = ['spaCy NER outputs:']
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc('Acme SmartWatch'), nlp.make_doc('Zeta Vacuum'), nlp.make_doc('SoundMax')]
    matcher.add('PRODUCT', patterns)
    for r in reviews:
        doc = nlp(r)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        matches = matcher(doc)
        found = [doc[start:end].text for _, start, end in matches]
        out_lines.append('Review: ' + r)
        out_lines.append('Entities: ' + str(ents))
        out_lines.append('Matched products/brands: ' + str(found))
        out_lines.append('')
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    text = '\n'.join(out_lines)
    ax.text(0,1, text, fontsize=10, family='monospace', va='top')
    plt.tight_layout()
    plt.savefig(os.path.join(SCREEN_DIR, 'spacy_ner.png'), dpi=150)
    plt.close()

# MNIST sample images
print('Generating MNIST sample images...')
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)
    idx = random.sample(range(len(X)), 5)
    images = X[idx].reshape(-1,28,28)
    fig, axs = plt.subplots(1,5, figsize=(10,2))
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(str(y[idx[i]]))
    plt.tight_layout()
    plt.savefig(os.path.join(SCREEN_DIR, 'mnist_samples.png'), dpi=150)
    plt.close()
except Exception as e:
    print('Failed to fetch MNIST:', e)
    # placeholder
    fig, ax = plt.subplots(figsize=(6,2))
    ax.axis('off')
    ax.text(0,0.5,'MNIST fetch failed. Please run in Colab to generate MNIST screenshots (GPU).', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(SCREEN_DIR, 'mnist_samples.png'), dpi=150)
    plt.close()

print('Screenshots generated in', SCREEN_DIR)

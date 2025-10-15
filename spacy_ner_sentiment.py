"""spaCy NER and simple rule-based sentiment demo
Run: python spacy_ner_sentiment.py
"""
import spacy
from spacy.matcher import PhraseMatcher


def main():
    # Load model (ensure installed: python -m spacy download en_core_web_sm)
    nlp = spacy.load('en_core_web_sm')

    reviews = [
        'I love the Acme SmartWatch, the battery lasts all week and the strap is comfortable.',
        'The Zeta Vacuum is noisy and stopped working after two weeks. Terrible experience.',
        'Great headphones by SoundMax — amazing bass and clear mids.'
    ]

    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc('Acme SmartWatch'), nlp.make_doc('Zeta Vacuum'), nlp.make_doc('SoundMax')]
    matcher.add('PRODUCT', patterns)

    pos_words = set(['love','great','amazing','good','excellent','comfortable'])
    neg_words = set(['terrible','noisy','bad','worst','stopped'])

    for r in reviews:
        doc = nlp(r)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        print('Review:', r)
        print('spaCy entities:', ents)
        matches = matcher(doc)
        found = [doc[start:end].text for _, start, end in matches]
        tokens = [t.text.lower() for t in doc]
        score = sum(1 for t in tokens if t in pos_words) - sum(1 for t in tokens if t in neg_words)
        sentiment = 'positive' if score>0 else ('negative' if score<0 else 'neutral')
        print('Matched products/brands:', found)
        print('Sentiment:', sentiment)
        print('---')


if __name__ == '__main__':
    main()

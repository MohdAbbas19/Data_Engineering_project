import pandas as pd
import nltk
nltk.download('vader_lexicon')
from textstat import flesch_kincaid_grade, syllable_count
from nltk.sentiment import SentimentIntensityAnalyzer
import re

output_df = pd.read_excel('Output_Data_Structure.xlsx')

def compute_variables(text):
    
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    

    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    polarity_score = sentiment_scores['compound']
    subjectivity_score = sentiment_scores['compound'] 
    
    avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if len(sentences) > 0 else 0
    
    num_complex_words = sum(1 for word in words if syllable_count(word) > 2)
    percentage_complex_words = (num_complex_words / len(words)) * 100 if len(words) > 0 else 0
    
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    avg_words_per_sentence = len(words) / len(sentences) if len(sentences) > 0 else 0
    
    complex_word_count = num_complex_words
    
    word_count = len(words)
    
    syllable_per_word = sum(syllable_count(word) for word in words) / len(words) if len(words) > 0 else 0
   
    personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    personal_pronoun_count = sum(1 for word in words if word.lower() in personal_pronouns)
    
    avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
    
    return [positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,
            complex_word_count, word_count, syllable_per_word, personal_pronoun_count, avg_word_length]

output_df[['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
           'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
           'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
           'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']] = output_df['ARTICLE TEXT'].apply(compute_variables).apply(pd.Series)
output_df.drop(['URL_ID', 'Title', 'Article_Text'], axis=1, inplace=True)
print(output_df)
output_df.to_csv('Output.csv', index=False)

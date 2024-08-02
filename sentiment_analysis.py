import pandas as pd                 # Data processing.
import numpy as np                  # Numerical analysis.
import re                           # Regular expression.
from textblob import TextBlob       # Free text processing library.
import seaborn as sns               # Data visualization Library.
import matplotlib.pyplot as plt     # Data visualization Library.
import pyfiglet

#Ascii printing Banner.
ascii_banner = pyfiglet.figlet_format("Sentiment Analysis")
print(ascii_banner)
ascii_banner = pyfiglet.figlet_format("--Byte Hotel--")
print(ascii_banner)

#Block to process the TEXT data.
df = pd.read_csv('tripadvisor_hotel_review.csv')
def clean_data(review):
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    return no_digits


# Perform SENTIMENT analysis using TextBlob.
df['Review'] = df['Review'].apply(clean_data)
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

df['Sentiment Polarity'], df['Sentiment Subjectivity'] = zip(*df['Review'].apply(get_sentiment))



# Interpret the sentiment POLARITY.
df['Sentiment'] = np.sign(df['Sentiment Polarity'])
df['Subjectivity'] = df['Sentiment Subjectivity'] * 5


# Visualization Of data and Printing.
sns.histplot(data=df, x='Sentiment', kde=True, color='b')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()
print(df)
print("#### We used a Criterial of -1 being Bad review, 0 being Nutral and +1 being Good ####")

#Print the data in a TXT file
with open('sentiment_analysis_output.txt', 'w') as f:
    f.write(df.to_string())
    f.write("\n\n#### We used a Criterial of -1 being Bad review, 0 being Nutral and +1 being Good ####")

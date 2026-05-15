# Code that scrapes text from Sinn Fein policy page, tokenises it, and creates a frequency dist 

# importing packages
from bs4 import BeautifulSoup
import requests
import pandas as pd 
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalProbDist
from nltk.probability import DictionaryProbDist
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm

# Downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Assigning name and email to html pull request 
headers = {
    "User-Agent": "prj(lucahollman@gmail.com)" 
}

# defining urls we are scraping

url = "https://housing.sinnfein.ie/vision/"

# Pulling html

r = requests.get(url, headers=headers)
output = BeautifulSoup(r.text, 'html.parser')

# Singling out data we need 

data = output.find('div', class_ = 'entry-content')
content = data.find_all('p')

# Tokenising data - seperating by word - and cleaning

text_t = word_tokenize(
    ' '.join(
        [value.text.strip() for value in content]
    ))

# defining stop words 

stop_words = set(stopwords.words("english"))

# Removing stop words from data

text_f = []
for w in text_t:
    if w not in stop_words:
        text_f.append(w)
        
# Creating frequency distribution 

fdist = nltk.FreqDist()
for word in text_f:
    fdist[word.lower()] += 1

del fdist["."]
del fdist[","]
del fdist["’"]
del fdist["%"]



fdist_w = fdist.most_common(15)

labels = [label[0] for label in fdist_w]



# Probability Dist

pdist = DictionaryProbDist(fdist, normalize=True)
print(pdist.prob("housing"))

df = pd.DataFrame({
    "words": labels,
    "freq": [frequency[1] for frequency in fdist_w ],
    "probability": [pdist.prob(word[0]) for word in fdist_w]
    })

print(df)




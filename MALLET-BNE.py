#!/usr/bin/env python
# coding: utf-8

# In[1]:


path_to_mallet = '/home/almahartwig/Downloads/mallet-2.0.8/bin/mallet'


# In[2]:


#!pip install little_mallet_wrapper
#!pip install seaborn
#!pip install nltk
#!pip install HanTa
#!pip install spacy-langdetect


# In[3]:


import little_mallet_wrapper
import seaborn
import glob
import pandas as pd
import numpy as np
import nltk
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


from nltk.corpus import stopwords


# In[4]:


directory = "/home/almahartwig/Dokumente/Studium-Master/SoSe-21/Hausarbeit-Trends-Info/Datensatz/BNE-Texte-txt"


# In[5]:


files = glob.glob(f"{directory}/*.txt")


# In[6]:


files


# In[7]:


stopwords = nltk.corpus.stopwords.words('german')
newStopWords = ['www','hrsg','pdf','bzgl','duk','for', 'abb', 'zb', 'bgl', 'bzw', 'ggf', 'etc', 'http']
stopwords.extend(newStopWords)
print(stopwords)


# In[8]:


little_mallet_wrapper.process_string('text', lowercase=True, numbers='remove', stop_words=stopwords)
#little_mallet_wrapper.process_string('text', lowercase=True, remove_short_words=True, remove_stop_words=True, remove_punctuation=True, numbers='remove', stop_words=stopwords, stop_words_extra)


# In[9]:


training_data = []
for file in files:
    text = open(file, encoding='utf-8').read()
    processed_text = little_mallet_wrapper.process_string(text, lowercase=True, numbers='remove', stop_words=stopwords)
    training_data.append(processed_text)
    
    firstElement = training_data[:1]

print(firstElement) 


# In[10]:



# Python3 code to demonstrate
# Tokenizing strings in list of strings
# using map() + split()
  
# initializing list
test_list = training_data
  
# printing original list
#print("The original list : " + str(test_list))
  
# using map() + split()
# Tokenizing strings in list of strings
res = list(map(str.split, test_list))
  
# print result
#print("The list after split of strings is : " + str(res))[:1]

firstElement = res[:1]

print(firstElement) 


# In[11]:


from HanTa import HanoverTagger as ht

tagger = ht.HanoverTagger('morphmodel_ger.pgz')

res = res

res_lemma = []
for word in res:
    lemma = [lemma for (word,lemma,pos) in tagger.tag_sent(word)]
    res_lemma.append(' '.join(lemma))


# In[12]:


firstElementRes = res_lemma[:1]

print(firstElementRes) 


# In[13]:


#check for HanTa
#print(tagger.analyze('nachhaltigkeitsthemen'))


# In[14]:


#hier alles in lowercase umwandeln 

strList = res_lemma

for i in range(len(strList)):
    strList[i] = strList[i].lower()
    
    firstElement_list = strList[:1]

print(firstElement_list)


# In[15]:


#hier noch mals Stopwörter entfernen; rem steht für removed

res_lemma_rem = []

for w in strList:
    if w not in stopwords:
        res_lemma_rem.append(w)
    
    firstElement_rlr = res_lemma_rem[:1]

print(firstElement_rlr) 

#language detection machen und englische Text evtl. rausschmeißen 


# In[16]:


original_texts = []
for file in files:
    text = open(file, encoding='utf-8').read()
    original_texts.append(text)


# In[17]:


obit_titles = [Path(file).stem for file in files]


# In[18]:


obit_titles


# In[19]:


#little_mallet_wrapper.print_dataset_stats(training_data)
little_mallet_wrapper.print_dataset_stats(res_lemma)


# In[20]:


num_topics = 17


# In[21]:


#training_data = training_data
#stemmed_training_data = stemmed_training_data

res_lemma_rem = res_lemma_rem


# In[22]:


output_directory_path = '/home/almahartwig/Dokumente/Studium-Master/SoSe-21/Hausarbeit-Trends-Info/Datensatz/BNE-texte-Egebnisse'


#No need to change anything below here
Path(f"{output_directory_path}").mkdir(parents=True, exist_ok=True)

path_to_training_data           = f"{output_directory_path}/training.txt"
path_to_formatted_training_data = f"{output_directory_path}/mallet.training"
path_to_model                   = f"{output_directory_path}/mallet.model.{str(num_topics)}"
path_to_topic_keys              = f"{output_directory_path}/mallet.topic_keys.{str(num_topics)}"
path_to_topic_distributions     = f"{output_directory_path}/mallet.topic_distributions.{str(num_topics)}"


# In[23]:


little_mallet_wrapper.quick_train_topic_model(path_to_mallet,
                                             output_directory_path,
                                             num_topics,
                                             res_lemma_rem)


# In[24]:


topics = little_mallet_wrapper.load_topic_keys(path_to_topic_keys)

for topic_number, topic in enumerate(topics):
    print(f"✨Topic {topic_number}✨\n\n{topic}\n")


# In[25]:


topic_distributions = little_mallet_wrapper.load_topic_distributions(path_to_topic_distributions)


# In[26]:


topic_distributions


# In[27]:


obituary_to_check = 'indikatoren-der-bildung-fuer-nachhaltige-entwicklung'

obit_number = obit_titles.index(obituary_to_check)

print(f"Topic Distributions for {obit_titles[obit_number]}\n")
for topic_number, (topic, topic_distribution) in enumerate(zip(topics, topic_distributions[obit_number])):
    print(f"✨Topic {topic_number} {topic[:6]} ✨\nProbability: {round(topic_distribution, 3)}\n")


# In[28]:


training_data_obit_titles = dict(zip(training_data, obit_titles))
training_data_original_text = dict(zip(training_data, original_texts))


# In[29]:


def display_top_titles_per_topic(topic_number=0, number_of_documents=5):
    
    print(f"✨Topic {topic_number}✨\n\n{topics[topic_number]}\n")

    for probability, document in little_mallet_wrapper.get_top_docs(training_data, topic_distributions, topic_number, n=number_of_documents):
        print(round(probability, 4), training_data_obit_titles[document] + "\n")
    return


# In[30]:


display_top_titles_per_topic(topic_number=0, number_of_documents=5)
display_top_titles_per_topic(topic_number=1, number_of_documents=5)
display_top_titles_per_topic(topic_number=2, number_of_documents=5)
display_top_titles_per_topic(topic_number=3, number_of_documents=5)
display_top_titles_per_topic(topic_number=4, number_of_documents=5)
display_top_titles_per_topic(topic_number=5, number_of_documents=5)
display_top_titles_per_topic(topic_number=6, number_of_documents=5)
display_top_titles_per_topic(topic_number=7, number_of_documents=5)
display_top_titles_per_topic(topic_number=8, number_of_documents=5)
display_top_titles_per_topic(topic_number=9, number_of_documents=5)
display_top_titles_per_topic(topic_number=10, number_of_documents=5)
display_top_titles_per_topic(topic_number=11, number_of_documents=5)
display_top_titles_per_topic(topic_number=12, number_of_documents=5)
display_top_titles_per_topic(topic_number=13, number_of_documents=5)
display_top_titles_per_topic(topic_number=14, number_of_documents=5)
display_top_titles_per_topic(topic_number=15, number_of_documents=5)
display_top_titles_per_topic(topic_number=16, number_of_documents=5)


# In[31]:


target_labels = ['indikatoren-der-bildung-fuer-nachhaltige-entwicklung',
 'philosophie-einer-humanen-bildung',
 'zukunftsfaehigkeit-im-kindergarten-vermitteln-kinder-staerken-nachhaltige-entwicklung-befoerdern',
 'unesco-heute-un-dekade-bne',
 'nachhaltigkeit-im-handel-n-tipps-fuer-die-ausbildungs-praxis',
 'nachhaltige-entwicklung-auf-kommunaler-ebene-durch-bildung-voranbringen',
 'gemeinsam-fuer-unsere-zukunft',
 'unesco-heute-nachhaltige-entwicklung',
 'qualitaetskriterien-fuer-die-fortbildung-von-multiplikatorinnen-und-multiplikatoren',
 'chancengerechtigkeit-im-deutschen-bildungssystem',
 'biologische-vielfalt-und-bildung-fuer-nachhaltige-entwicklung',
 'hochschulen-fuer-nachhaltige-entwicklung-erklaerung-der-hochschulrektorenkonferenz-und-der-deutschen-unesco-kommission-zur-hochschulbildung-fuer-nachhaltige-entwicklung',
 'hochschulen-fuer-eine-nachhaltige-entwicklung-ideen-zur-institutionalisierung-und-implementierung',
 'hochschulen-fuer-eine-nachhaltige-entwicklung-nachhaltigkeit-in-forschung-lehre-und-betrieb',
 'umsetzung-der-un-dekade-bildung-fuer-nachhaltige-entwicklung-zwischenbericht-mitte-2007-bis-mitte-2010',
 'ausgezeichnet-kommunen-lernorte-und-netzwerke',
 'vom-projekt-zur-struktur-projekte-massnahmen-und-kommunen-der-un-dekade-bildung-fuer-nachhaltige-entwicklung',
 'bonner-erklaerung-2014',
 'querbeet-biologische-vielfalt-und-bildung-fuer-nachhaltige-entwicklung-anregungen-fuer-die-praxis',
 'informationsblatt-zur-einreichung-von-commitments-zum-nationalen-aktionsplan-bildung-fuer-nachhaltige-entwicklung-nap-bne',
 'bildung-fuer-nachhaltige-entwicklung-im-spiegel-von-kunst-kultur-und-kultureller-bildung',
 'hochschulen-fuer-eine-nachhaltige-entwicklung-netzwerke-foerdern-bewusstsein-verbreiten',
 'nationaler-aktionsplan-bildung-fuer-nachhaltige-entwicklung-2017',
 'der-beitrag-der-un-dekade-2005-2014-zu-verbreitung-und-verankerung-der-bildung-fuer-nachhaltige-entwicklung',
 'un-dekade-mit-wirkung-10-jahre-bne-in-deutschland',
 'zukunftsstrategie-2015-positionspapier-des-nationalkomitees-der-un-dekade',
 'vom-projekt-zur-struktur-strategiepapier-der-ag-berufliche-aus-und-weiterbildung',
 'lehrmaterialien-zum-jahresthema-mobilitaet',
 'deutsche-unesco-kommission-jahrbuch-2016-2017',
 'roadmap-zur-umsetzung-des-weltaktionsprogramms',
 'zukunftsfaehige-kommunen-chancen-durch-bildung-fuer-nachhaltige-entwicklung',
 'tagungsbericht-un-dekade-bildung-fuer-nachhaltige-entwicklung-der-beitrag-europas',
 'zwischenbilanz-zum-nationalen-aktionsplan-bildung-fuer-nachhaltige-entwicklung',
 'strukturen-staerken-ausgezeichnete-kommunen-lernorte-und-netzwerke-des-unesco-weltaktionsprogramms-bildung-fuer-nachhaltige-entwicklung-im-portraet',
 'referenzrahmen-fuer-die-fruehkindliche-bildung']


# In[32]:


little_mallet_wrapper.plot_categories_by_topics_heatmap(obit_titles,
                                      topic_distributions,
                                      topics, 
                                      output_directory_path + '/categories_by_topics.png',
                                      target_labels=target_labels,
                                      dim= (16, 9)
                                     )


# In[33]:


from IPython.display import Markdown, display
import re

def display_bolded_topic_words_in_context(topic_number=3, number_of_documents=3, custom_words=None):

    for probability, document in little_mallet_wrapper.get_top_docs(training_data, topic_distributions, topic_number, n=number_of_documents):
        
        print(f"✨Topic {topic_number}✨\n\n{topics[topic_number]}\n")
        
        probability = f"✨✨✨\n\n**{probability}**"
        obit_title = f"**{training_data_obit_titles[document]}**"
        original_text = training_data_original_text[document]
        topic_words = topics[topic_number]
        topic_words = custom_words if custom_words != None else topic_words

        for word in topic_words:
            if word in original_text:
                original_text = re.sub(f"\\b{word}\\b", f"**{word}**", original_text)

        display(Markdown(probability)), display(Markdown(obit_title)), display(Markdown(original_text))
    return


# In[34]:


display_bolded_topic_words_in_context(topic_number=1, number_of_documents=3)


# In[ ]:





import pandas as pd 
import numpy as np
import nltk 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval



#nltk doğal dil işleme 
data = pd.read_csv("Hotel_Reviews.csv")

print(data.head(5))


#birleşik krallık yerine UK

data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom","UK")
#adresi bölelim ve ülkeyi tanımlamak için son kelimeyi seçelim
data["countries"] = data.Hotel_Address.apply(lambda x : x.split(' ')[-1])
print(data.countries.unique())


#şimdi işimize yaramayacak özellikleri silelim

data.drop(['Additional_Number_of_Scoring', 'Review_Date', 'Reviewer_Nationality',
           'Negative_Review', 'Review_Total_Negative_Word_Counts',
           'Total_Number_of_Reviews', 'Positive_Review',
           'Review_Total_Positive_Word_Counts',
           'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score',
           'days_since_review', 'lat', 'lng'], axis=1, inplace=True)


#Şimdi listenin dizelerini normal bir listeye dönüştürecek ve daha sonra onu veri kümesindeki “Etiketler” sütununa uygulayacak bir fonksiyon oluşturacağım:


def impute(column):
    column = column[0]
    if (type(column) != list):
        return "".join(literal_eval(column))
    else:
        return column
    
data["Tags"] = data[["Tags"]].apply(impute, axis=1)
data.head()


#Şimdi basitlik açısından "Etiketler" ve "ülkeler" sütununu küçük harfle yazacağım:

data['countries'] = data['countries'].str.lower()
data['Tags'] = data['Tags'].str.lower()



#Şimdi lokasyona ve kullanıcının verdiği açıklamaya göre otel isimlerini önerecek bir fonksiyon tanımlayalım. Burada amacımız sadece otelin adını tavsiye etmek değil, aynı zamanda kullanıcı derecelendirmelerine göre de sıralamaktır:

    
def recommend_hotel(location,description):
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
        
    country = data[data['countries']==location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    list1 = [];list2 = [];cos = [];
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    country['similarity'] = cos 
    country = country.sort_values(by = 'similarity',ascending = False)
    country.drop_duplicates(subset = 'Hotel_Name',keep = 'first',inplace = True)
    country.sort_values('Avarage_Score',ascending = False ,inplace = True)
    country.rese_index(inplace = True)
    return country[["Hotel_Name", "Average_Score", "Hotel_Address"]].head()
    

    
recommend_hotel('Italy', 'I am going for a business trip')




***********************************
import pandas as pd 
import numpy as np
import nltk 
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

# NLTK doğal dil işleme
data = pd.read_csv("Hotel_Reviews.csv")

print(data.head(5))

# Birleşik Krallık yerine UK
data.Hotel_Address = data.Hotel_Address.str.replace("United Kingdom", "UK")
# Adresi bölelim ve ülkeyi tanımlamak için adresteki son kelimeyi seçelim
data["countries"] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
print(data.countries.unique())

# Şimdi işimize yaramayacak özellikleri silelim
data.drop(['Additional_Number_of_Scoring', 'Review_Date', 'Reviewer_Nationality',
           'Negative_Review', 'Review_Total_Negative_Word_Counts',
           'Total_Number_of_Reviews', 'Positive_Review',
           'Review_Total_Positive_Word_Counts',
           'Total_Number_of_Reviews_Reviewer_Has_Given',
           'days_since_review', 'lat', 'lng'], axis=1, inplace=True)

# Şimdi listenin dizelerini normal bir listeye dönüştürecek ve daha sonra onu veri kümesindeki "Etiketler" sütununa uygulayacak bir fonksiyon oluşturacağım:
def impute(column):
    column = column.iloc[0]  # column[0] yerine column.iloc[0] kullanılıyor
    if type(column) != list:
        return "".join(literal_eval(column))
    else:
        return column


data["Tags"] = data[["Tags"]].apply(impute, axis=1)
data.head()

# Şimdi basitlik açısından "Etiketler" ve "ülkeler" sütununu küçük harfle yazacağım:
data['countries'] = data['countries'].str.lower()
data['Tags'] = data['Tags'].str.lower()

# Şimdi lokasyona ve kullanıcının verdiği açıklamaya göre otel isimlerini önerecek bir fonksiyon tanımlayalım. 
# Burada amacımız sadece otelin adını tavsiye etmek değil, aynı zamanda kullanıcı derecelendirmelerine göre de sıralamaktır:
def recommend_hotel(location, description):
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
        
    country = data[data['countries'] == location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    list1 = []; list2 = []; cos = []
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    country['similarity'] = cos 
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Reviewer_Score', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    return country[["Hotel_Name", "Reviewer_Score", "Hotel_Address"]].head()

recommend_hotel('Italy', 'I am going for a business trip')
recommend_hotel('UK','I am going on a honeymoon, I need a honeymoon suite room for 3 nights')



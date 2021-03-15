import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import RFE
from core.utils import UtilMethods


class BOCW(UtilMethods):

    def __init__(self, file_path, output_folder, text_column, class_column,
                 cleaner=None, n_words=20):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1].split('.')[0]
        self.output_folder = output_folder
        self.output_path = f'{output_folder}/{self.file_name}'
        self.create_folder(self.output_path)
        self.text_col = text_column
        self.class_col = class_column
        if cleaner is None:
            self.clean_sentence = self.default_cleaner
        else:
            self.clean_sentence = cleaner
        self.n_words = n_words
        self.keep_n_grams = False
        self.frame = pd.read_csv(file_path)
        self.terms = {}

    @staticmethod
    def default_cleaner(sentence, as_string=False):
        sentence = sentence.lower()
        stopwords_english = stopwords.words('english')
        sentence = re.sub(r'\$\w*', '', sentence)
        sentence = re.sub(r'^RT[\s]+', '', sentence)
        sentence = re.sub(r'https?:.*[\r\n]*', '', sentence)
        sentence = re.sub(r'#', '', sentence)
        tokens = word_tokenize(text=sentence)
        sentence_clean = []
        lemming = WordNetLemmatizer()
        for word in tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation and word != 'rt'
                    and word != '...' and not word.isdigit()
                    and word.isalpha()):
                stem_word = lemming.lemmatize(word, pos="v")
                sentence_clean.append(unidecode.unidecode(stem_word))
        if as_string:
            return ' '.join(sentence_clean)
        else:
            return sentence_clean

    def generate_n_grams(self, sentences, n_grams):
        words = []
        for sentence in tqdm(sentences, desc="cleaning sentence"):
            clean = self.clean_sentence(sentence)
            final_group = []
            for window in range(len(clean)):
                subgroup = clean[window:window + n_grams]
                subgroup = ' '.join(subgroup)
                final_group.append(subgroup)
            words.append(final_group)
        words = sum(words, [])
        return words

    def __extract_most_common_words_by_class(self, list_of_words, class_value):
        word_freq = FreqDist(list_of_words)
        df = pd.DataFrame(
            {f'{class_value}_words': list(word_freq.keys()), f'{class_value}_counts': list(word_freq.values())})
        df = df.nlargest(self.n_words, columns=f'{class_value}_counts')
        df.reset_index(drop=True, inplace=True)
        return df

    def __extract_n_gram_by_class(self, class_value, n_gram=1):
        list_of_words = self.generate_n_grams(self.frame[self.text_col]
                                              [self.frame[self.class_col] == class_value], n_gram)
        n_gram_frame = self.__extract_most_common_words_by_class(list_of_words, class_value)

        return n_gram_frame

    def extract_critical_words_by_gram(self, n_gram=1, keep_n_grams=True):
        frame_base = pd.DataFrame({})
        for value in self.frame[self.class_col].unique():
            sentiment_frame = self.__extract_n_gram_by_class(value, n_gram)
            frame_base = pd.concat([frame_base, sentiment_frame], axis=1)
        if keep_n_grams:
            frame_base.to_csv(f'{self.output_path}/results_{n_gram}_grams.csv', index=False)
        return frame_base

    def extract_critical_words(self, grams, majority_class, keep=True, unique_f=True, max_words=5):
        critical_words = []
        for gram in grams:
            frame = self.extract_critical_words_by_gram(n_gram=gram, keep_n_grams=keep)
            i = 0
            for n, row in frame.iterrows():
                target_value = row[2]
                target_counts = row[3]
                if unique_f:
                    if target_value not in list(frame[f"{majority_class}_words"]) and i < max_words:
                        critical_words.append(target_value)
                        i += 1
                    else:
                        if i < max_words:
                            index_location = frame[frame[f'{majority_class}_words'] == target_value].index.item()
                            real_counts = frame.at[index_location, f'{majority_class}_counts']
                            if target_counts > real_counts:
                                critical_words.append(target_value)
                                i += 1
                else:
                    if i < max_words:
                        critical_words.append(target_value)
                        i += 1
        return critical_words

    def get_counts(self, tweet):
        for key in self.terms.keys():
            self.terms[key].append(int(key.lower() in tweet.lower()))

    def extract(self, grams, majority_class, terms=None, keep=True, unique_f=True, max_words=5):
        if terms is None:
            words = self.extract_critical_words(grams, majority_class, keep, unique_f, max_words)
            self.terms = {term: [] for term in words}
        else:
            self.terms = terms

        for text in tqdm(self.frame[self.text_col], desc="loading sentence"):
            text = ' '.join(self.default_cleaner(text))
            self.get_counts(text)

        frame = pd.DataFrame(self.terms)
        frame[self.class_col] = self.frame[self.class_col]
        frame.to_csv(f'{self.output_path}/BOCW.csv', index=False)
        features = self.rfe(frame)
        frame = frame[features+[self.class_col]]
        frame.to_csv(f'{self.output_path}/BOCW_RFE.csv', index=False)

    def rfe(self, frame):
        if frame[self.class_col].dtype == "object":
            frame[self.class_col] = frame[self.class_col].astype('category')
            frame[self.class_col] = frame[self.class_col].cat.codes

        Y = frame[[self.class_col]]
        Y = Y.fillna(0)
        X = frame.drop(columns=[self.class_col])
        nof_list = np.arange(1, len(X.columns) + 1)
        high_score = 0
        nof = 0
        score_list = []
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
            model = LinearRegression()
            rfe = RFE(model, n_features_to_select=nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train, y_train.values.ravel())
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe, y_train)
            score = model.score(X_test_rfe, y_test)
            score_list.append(score)
            if score > high_score:
                high_score = score
                nof = nof_list[n]
        cols = list(X.columns)
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=nof)
        X_rfe = rfe.fit_transform(X, Y.values.ravel())
        model.fit(X_rfe, Y.values.ravel())
        temp = pd.Series(rfe.support_, index=cols)
        selected_features_rfe = list(temp[temp == True].index)
        return selected_features_rfe

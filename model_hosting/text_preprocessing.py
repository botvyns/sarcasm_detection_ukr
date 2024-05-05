from nltk.tokenize import TweetTokenizer
import stanza
import re

tk = TweetTokenizer()
uk_nlp = stanza.Pipeline(lang='uk', verbose=False)

def substitute_user_mentions_and_links(text):
    # Regular expression to match user mentions (e.g., @username)
    user_mention_pattern = r'@\w+'

    # Regular expression to match links (e.g., http://example.com)
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Substitute user mentions 
    text = re.sub(user_mention_pattern, '', text)

    # Substitute links 
    text = re.sub(link_pattern, '', text)

    # Substitute latin chars
    text = re.sub(r'[a-zA-Z]+', '', text)

    return text.lower()

def remove_some_punc_numbers(text):
    chars_to_remove = r'[\#\$\%\&\*\+\,\-\/\:\;\<\=\>\@\[\\\]\^\_\{\|\}\~\d\.\–]'

    result = re.sub(chars_to_remove, '', ' '.join(text))

    return result.lower()

pattern = r'\b(\w+)\s*\'\s*(\w+)\b'

# Define a function to join words separated by single quotes
def join_words(match):
    return match.group(1) + "'" + match.group(2)

def lemmatize(text):
    lemmas_st = []
    for sent in uk_nlp(text).sentences:
        for word in sent.words:
            lemmas_st.append(word.lemma)
    return lemmas_st

def preprocess_text(input_text):

    text_mod = substitute_user_mentions_and_links(input_text)
    tokenized = tk.tokenize(text_mod)
    spec_char_remv = remove_some_punc_numbers(tokenized)
    apostrophe_fixed = re.sub(pattern, join_words, spec_char_remv)
    spaces_fixed = re.sub(r'\s+', ' ', apostrophe_fixed)
    lemmatized = lemmatize(spaces_fixed)

    return text_mod, lemmatized

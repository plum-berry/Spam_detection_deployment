import re
import numpy as np
import pandas as pd
SPAMBASE_WORDS = [
    'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
    'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
    'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
    'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857',
    'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct',
    'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference'
]

SPAMBASE_CHARS = [';', '(', '[', '!', '$', '#']


def extractFeatures(text):
    text_lower =  text.lower()
    words = re.findall(r'\b\w+\b',text_lower)
    total_words = len(words)

    word_freqs = []
    for word in SPAMBASE_WORDS:
        count = words.count(word)
        freq = (count/total_words)*100
        word_freqs.append(freq)

    total_chars = len(text)
    char_freqs = []
    for char in SPAMBASE_CHARS:
        count = text.count(char)
        freq = (count/total_chars)
        char_freqs.append(freq)


    capital_runs = re.findall(r'[A-Z]{2,}',text)
    if capital_runs:
        run_lengths = [len(run) for run in capital_runs]
        capital_run_avg = np.mean(run_lengths)
        capital_run_longest = np.max(run_lengths)
        capital_run_sum =  np.sum(run_lengths)
    else:
        capital_run_avg = 0
        capital_run_longest = 0
        capital_run_sum =  0

    features = word_freqs + char_freqs + [capital_run_avg, capital_run_longest, capital_run_sum]
    feature_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
    feat_arr = np.array(features).reshape(1,-1)
    feature_df = pd.DataFrame(feat_arr,columns=feature_names)
    return feature_df


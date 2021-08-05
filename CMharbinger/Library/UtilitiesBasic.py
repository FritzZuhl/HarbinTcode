# Base character handling functions
###
import pandas
import collections
# import nltk

def flatten(list_of_lists):
    '''
    Flattens a list of lists. Works best when all lists have strings for elements.
    :param list_of_lists: a list of strings or many-layered list of strings.
    :return: a flatten list
    :warning: input cannot be too deep, else it will trigger recursion limit error.
    '''
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def count_tokens_out_df(data):
    '''
    Counts the tokens in a list. Makes no assumptions about the tokenization
    :param data: a list made of tokens, usually tokenized from a document
    :return: a dataframe with words and their counts
    '''
    list_of_tokens = data
    try:
        counts = collections.Counter(list_of_tokens)
    except:
        print("Cannot count input.")
        exit()
    counts_d = dict(counts)
    counts_df = pandas.DataFrame.from_dict(counts_d, orient='index',
                                           columns=['count'])
    counts_df.reset_index(inplace=True)
    counts_df = counts_df.rename(columns={'index':'word'})
    counts_df = counts_df.sort_values("count", ascending=False)
    counts_df.reset_index(inplace=True, drop=True)
    return counts_df


# Base character handling functions
###
import pandas
import collections
# import nltk

# A good list flattener when dealing with lists that either have text, or deeper lists.
# Cannot be used on super deep lists, because you can run into recursion limits.
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def count_tokens_out_df(data):
    list_of_strings = data
    try:
        counts = collections.Counter(list_of_strings)
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


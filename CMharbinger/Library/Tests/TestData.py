import pandas


# for testing:
# UtilitiesBasic.flatten
list1Test = ['a', ['b', 'c']]
list1Expect = ['a', 'b', 'c']
#
list2Test = ['a', 'b', 'c']
list2Expect = ['a', 'b', 'c']

# for testing:
# UtilitiesBasic.count_tokens_out_df
doc2 = ['a', 'b', 'c', 'b', 'c']
doc2_Expect = pandas.DataFrame({'word':['b','c','a'], 'count':[2,2,1]})

doc1 = "Now is the time for all good men to come to the aid of their country."
word_count_doc1 = 16

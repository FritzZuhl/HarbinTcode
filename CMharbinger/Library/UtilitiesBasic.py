# Base character handling functions
###


# A good list flattener when dealing with lists that either have text, or deeper lists.
# Cannot be used on super deep lists, because you can run into recursion limits.
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


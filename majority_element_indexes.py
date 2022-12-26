def majority_element_indices(lst):
    '''
    return a list of indexes of the majority element if it exists.
    Majority element is the element that appears more than floor(N/2) times.
    If there is no majority element, return []
    >>> majority_element_indices([1,1,2])
    [0, 1]
    >>> majority_element_indices([1,2])
    []
    >>> majority_element_indices([])
    []

    '''
    from collections import Counter
    from math import floor
    lst_len = len(lst)
    if lst_len < 2:  # trivial list
        return []

    lst_cnts = Counter(lst)
    most_commen = lst_cnts.most_common(1)[0][0]
    occures = lst_cnts.most_common(1)[0][1]

    if floor(lst_len/2) >= occures:  # there is no majority element
        return []

    indexes = [index for index,value in enumerate(lst) if value == most_commen]
    return indexes

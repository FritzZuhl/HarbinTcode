a,b,c = input("Enter the value for a, b, c :").split()
print(a)
print(b)
print(c)


import itertools
name = 'Python'
for i in itertools.permutations(name):
    print(i)


matrix=[[1,2,44],[3,4,55],[5,6,66]]
trans=zip( *matrix)
print(list(trans))
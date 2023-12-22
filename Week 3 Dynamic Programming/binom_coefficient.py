memoizedTable = {} # This is a Python dictionary
global counter 
counter = 0
def binomial_coefficient(n, k):
    assert n >= k
    key = (n, k)
    print(key)
    if k == 0 or k == n:
        return 1
    if key not in memoizedTable:
        memoizedTable[(n,k)] = binomial_coefficient(n-1, k) + binomial_coefficient(n-1, k-1)
        return memoizedTable[(n,k)]
    else: 
        return memoizedTable[(n,k)]


print(binomial_coefficient(4,2))
print(len(memoizedTable))
# values[] stores the values for each item
# weights[] stores the weights for each item
# n is the number of items
# C is the maximum capacity of bag
# table[C+1] to store final result
def knapSack(values, weights, n, C):
    table = [0]*(C+1);
    for i in range(n):
        for j in range(C,weights[i]-1,-1):
            table[j] = max(table[j] , values[i] + table[j-weights[i]]);
    return table[C];
 
 
# Driver program to test the cases
values = [7, 8, 4];
weights = [3, 8, 6];
C = 10; 
n = 3;
print(knapSack(values, weights, n, C));
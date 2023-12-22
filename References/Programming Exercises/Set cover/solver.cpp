#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <new>
#include <algorithm>
#include <stack>
#include <map>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <random>
#include <getopt.h>
#include <omp.h>
#include <immintrin.h>
#include <bits/stdc++.h>
#include <vector>
#include <queue>
#include <utility>

using namespace std;

// This function return the difference between the number of uncover elements and the number of uncover element of one set.
int uncover_diff(bool *uncover, pair<int, int> top, const int *p, const int *f)
{
    auto set_index = top.second;
    int uncover_elements = 0;
    for(int i = p[set_index]; i < p[set_index + 1]; i++) {
        if(uncover[f[i]]) {
            uncover_elements ++;
        }    
    }
    return (top.first - uncover_elements);
}

void solver(int n, int m, const int *p, const int *f, int &k, int *r)
{
    priority_queue<pair<int, int>> sets_queue; // The first element in the pair is the number of uncovered element, and the second number is the set index
    bool *uncover = new bool[n];                      // uncover[i] is true if i is not covered yet
    int covered = 0;
    k = 0;
    for(int i = 0; i < n; i++) {
        uncover[i] = true;
    }
    for(int i = 0; i < m; i++) {
        sets_queue.push(make_pair((p[i+1] - p[i]), i));
    }
    while(covered != n) {
        if(sets_queue.empty()) {
            k = 0;
            return;
        }
        auto curr_set = sets_queue.top(); // The best set
        sets_queue.pop();
        int curr_set_index = curr_set.second; 
        r[k] = curr_set_index;
        k ++;
        for(int i = p[curr_set_index]; i < p[curr_set_index + 1]; i++) {           // Loop through all element in the chosen set. For each element, find whether that element is in the uncover vector, remove if it does
            int element = f[i];
            if(uncover[element]) {
                uncover[element] = false;
                covered ++;
            }
        }
        auto next_set = sets_queue.top();
        int diff = uncover_diff(uncover, next_set, p, f); 
        while(diff != 0) {                // Loop while the first element in the queue is not up to date
            sets_queue.pop();
            sets_queue.push(make_pair((next_set.first - diff), next_set.second));
            next_set = sets_queue.top();
            diff = uncover_diff(uncover, next_set, p, f);
        }
    }
    delete[] uncover;
    return;
}
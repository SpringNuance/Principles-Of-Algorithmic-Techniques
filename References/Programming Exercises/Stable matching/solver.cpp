void solver(int n, const int *a_pref, const int *b_pref, int *s)
{
    int fill = 0;
    // The order of each application for each company (i.e., where does applicant number c falls in the preference of company y)
    int *position = new int [n];
    perm_inv(n, b_pref, position);
    // Use to keep track of which companies/ applicant is matched
    bool *matched = new bool[n];
    for(int f = 0; f <= n; f ++) {
        matched[f] = false;
    }
    bool *p_matched = new bool[n];
    for(int f = 0; f <= n; f ++) {
        p_matched[f] = false;
    }
    // Loop while the number of filled applicant == n
    while(fill < n) {
        for(int m = 0; m < n; m ++) {
            // Loop for all applicants
            if(!matched[m]) {
                for(int f = 0; f < n; f ++) {
                    // If the company is unmatched
                    if(!p_matched[a_pref[m*n+f]]) {
                        s[m] = a_pref[m*n+f];
                        matched[m] = true;
                        p_matched[m*n+f] = true;  
                        fill += 1;
                        break;
                    // If the company has an applicant
                    } else {
                        // Get the applicant of the company and compare its position to the current applicant
                        int *inverse = new int[n];
                        perm_inv(n, s, inverse);
                        if(position[inverse[m*n+f]] < position[m]) {
                            matched[inverse[m*n+f]] = false;
                            s[m] = m*n+f;
                            matched[m] = true;
                        }
                        delete[] inverse;
                        break;
                    }
                }
            }
        }
    }
    delete[] position;
    delete[] matched;
    delete[] p_matched;
    return;
}
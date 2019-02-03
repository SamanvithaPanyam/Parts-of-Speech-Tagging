import numpy as np
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    max_score = 0
    y = []
    scores = []
    indices = []
    for i in xrange(N):
        if i == 0:
            first_row_scores = []
            temp_indices = []
            for j in range(L):
                first_row_scores.append(start_scores[j] + emission_scores[i][j])
                temp_indices.append(0)
            scores.append(first_row_scores)
            indices.append(temp_indices)

        else:
            row = []
            temp_indices = []
            for j in xrange(L):
                mid_row_scores = []
                for k in xrange(L):
                    mid_row_scores.append(scores[i-1][k] + trans_scores[k][j] + emission_scores[i][j])
                row.append(max(mid_row_scores))
                temp_indices.append(np.argmax(mid_row_scores))
            scores.append(row)
            indices.append(temp_indices)

        if i == N-1:
            if N != 1:
                for j in xrange(L):
                    scores[i][j] = scores[i][j] + end_scores[j]
            else:
                for j in xrange(L):
                    scores[0][j] = scores[0][j] + end_scores[j]
            max_score = np.max(scores[-1])

    max_index = np.argmax(scores[-1])
    y.append(max_index)  
    temp_val = N-1
    while temp_val > 0:
            max_index = indices[temp_val][max_index]
            y.append(max_index)
            temp_val = temp_val - 1
    y.reverse()




    return (max_score, np.asarray(y))

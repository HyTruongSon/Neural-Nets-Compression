#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <ctime>

#include "mex.h"

using namespace std;

const int maxIters = 100;

// nRows: number of rows
// nCols: number of columns
int nRows, nCols;

// Number of clusters
int k;

// Original matrix
double **A;

// Quantized matrix
double **B;

// Cell
struct Cell {
    double value;
    int row, col;
    
    bool operator < (const Cell &another) {
        if (value < another.value) {
            return true;
        }
        return false;
    }
};
Cell *C;

// Cluster
struct Cluster {
    double value;
    int start, finish;
};
Cluster *D;

// Sum
double *S;

void vector2matrix(double *input, int nRows, int nCols, double **output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[i][j] = input[j * nRows + i];
        }
    }
}

void matrix2vector(double **input, int nRows, int nCols, double *output) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            output[j * nRows + i] = input[i][j];
        }
    }
}

double find_sum(int start, int finish) {
    if (start == 0) {
        return S[finish];
    }
    return S[finish] - S[start - 1];
}

double find_average(int start, int finish) {
    return find_sum(start, finish) / (finish - start + 1);
}

void mexFunction(int nOutputs, mxArray *output_pointers[], int nInputs, const mxArray *input_pointers[]) {
    if (nInputs != 2) {
        std::cerr << "The number of input parameters must be exactly 2!" << std::endl;
        return;
    }
    
    // nRows: number of rows
    // nCols: number of columns
    nRows = mxGetM(input_pointers[0]);
    nCols = mxGetN(input_pointers[0]);
    
    // std::cout << "Number of rows: " << nRows << endl;
    // std::cout << "Number of columns: " << nCols << endl;
    
    // Number of clusters
    k = mxGetScalar(input_pointers[1]);
    
    // std::cout << "Number of clusters: " << k << endl;
    
    // Memory allocation
    A = new double* [nRows];
    B = new double* [nRows];
    C = new Cell [nRows * nCols];
    D = new Cluster [k];
    S = new double [nRows * nCols];
    
    for (int row = 0; row < nRows; ++row) {
        A[row] = new double [nCols];
        B[row] = new double [nCols];
    }
    
    vector2matrix(mxGetPr(input_pointers[0]), nRows, nCols, A);
    
    for (int row = 0; row < nRows; ++row) {
        for (int col = 0; col < nCols; ++col) {
            int index = row * nCols + col;
            C[index].row = row;
            C[index].col = col;
            C[index].value = A[row][col];
        }
    }
    
    sort(C, C + (nRows * nCols));
    
    S[0] = C[0].value;
    for (int i = 1; i < nRows * nCols; ++i) {
        S[i] = S[i - 1] + C[i].value;
    }
    
    int average_length = nRows * nCols / k;
    D[0].start = 0;
    for (int i = 0; i < k; ++i) {
        D[i].finish = D[i].start + average_length - 1;
        if (i + 1 < k) {
            D[i + 1].start = D[i].finish + 1;
        }
    }
    D[k - 1].finish = nRows * nCols - 1;
    
    for (int i = 0; i < k; ++i) {
        D[i].value = find_average(D[i].start, D[i].finish);
    }
    
    for (int iter = 0; iter < maxIters; ++iter) {
        for (int i = 1; i < k; ++i) {
            int position = D[i].finish;
            int left = D[i - 1].start;
            int right = D[i].finish;
            while (left <= right) {
                int mid = (left + right) / 2;
                if (abs(C[mid].value - D[i - 1].value) >= abs(C[mid].value - D[i].value)) {
                    position = min(position, mid);
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            D[i - 1].finish = position - 1;
            D[i].start = position;
            D[i - 1].value = find_average(D[i - 1].start, D[i - 1].finish);
            D[i].value = find_average(D[i].start, D[i].finish);
        }
    }
    
    for (int i = 0; i < k; ++i) {
        for (int j = D[i].start; j <= D[i].finish; ++j) {
            B[C[j].row][C[j].col] = D[i].value;
        }
    }
    
    output_pointers[0] = mxCreateDoubleMatrix(nRows, nCols, mxREAL);
    matrix2vector(B, nRows, nCols, mxGetPr(output_pointers[0]));
    
    for (int row = 0; row < nRows; ++row) {
        delete A[row];
        delete B[row];
    }
    delete C;
    delete D;
    delete S;
}
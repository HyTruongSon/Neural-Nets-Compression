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

vector<string> words_separation(string line) {
    vector<string> words;
    words.clear();

    int i = 0;
    while (i < line.length()) {
        if (isspace(line[i])) {
            ++i;
            continue;
        }
        string word = "";
        for (int j = i; j < line.length(); ++j) {
            if (!isspace(line[j])) {
                i = j;
                word += line[j];
            } else {
                break;
            }
        }
        ++i;
        words.push_back(word);
    }
    return words;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "The number of input parameters must be exactly 3!" << std::endl;
        return 0;
    }

    char *input_fn = argv[1];
    char *output_fn = argv[2];
    
    ifstream input(input_fn, ios::in);
    string line;
    nRows = 0;
    vector<string> lines;
    lines.clear();
    while (true) {
        if (!getline(input, line)) {
            break;
        }
        if (line.length() == 0) {
            continue;
        }
        ++nRows;
        lines.push_back(line);
    }
    input.close();

    vector<string> words = words_separation(lines[0]);
    nCols = words.size();

    std::cout << "Number of rows: " << nRows << endl;
    std::cout << "Number of columns: " << nCols << endl;
    
    // Number of clusters
    k = atoi(argv[3]);
    
    std::cout << "Number of clusters: " << k << endl;
    
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
    
    for (int row = 0; row < nRows; ++row) {
        words = words_separation(lines[row]);
        for (int col = 0; col < nCols; ++col) {
            A[row][col] = atof(words[col].c_str());
        }
    }
    
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
    
    ofstream output(output_fn, ios::out);
    for (int row = 0; row < nRows; ++row) {
        for (int col = 0; col < nCols; ++col) {
            output << B[row][col] << " ";
        }
        output << endl;
    }
    output.close();
    
    for (int row = 0; row < nRows; ++row) {
        delete A[row];
        delete B[row];
    }
    delete C;
    delete D;
    delete S;

    return 0;
}
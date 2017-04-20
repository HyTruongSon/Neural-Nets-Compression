// Software: Training Autoencoder (Neural Network) for MNIST database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2015 (c). All rights reserved.

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// Dataset file names
string training_image_fn = "MNIST/train-images.idx3-ubyte";
string testing_image_fn = "MNIST/t10k-images.idx3-ubyte";
string text_training_image_fn = "saves/train-images-colors.dat";
string text_testing_image_fn = "saves/test-images-colors.dat";

// Number of hidden layers
const int nHidden = 256;

// Number of training samples
const int nTraining = 60000;

// Number of testing samples
const int nTesting = 10000;

// Weights file name
const string model_fn = "saves/model-Autoencoder-nHidden-256.dat";

// Image size in MNIST database
const int width = 28;
const int height = 28;
const int nInput = width * height;

const int Epochs = 10;
const int nIterations = 30;

const double learning_rate = 1e-2;
const double momentum = 0.9;
const double epsilon = 5.0;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double **W, **delta1, *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double **delta2, *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double *expected;

// Image. In MNIST: 28x28 gray scale images.
double d[width][height];

// File stream to read data
ifstream image;

// Show images or not
bool showImages;

// +--------------------+
// | About the software |
// +--------------------+

void about() {
	cout << "***********************************************" << endl;
	cout << "*** Training Autoencoder for MNIST database ***" << endl;
	cout << "***********************************************" << endl;
	cout << endl;
	cout << "Number of input neurons: " << nInput << endl;
	cout << "Number of hidden neurons: " << nHidden << endl;
	cout << endl;
    cout << "Number of epochs: " << Epochs << endl;
	cout << "Number of iterations: " << nIterations << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
    cout << "Testing image data: " << testing_image_fn << endl;
	cout << "Number of training samples: " << nTraining << endl;
    cout << "Number of testing samples: " << nTesting << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    W = new double* [nInput];
    delta1 = new double* [nInput];

    for (int i = 0; i < nInput; ++i) {
        W[i] = new double [nHidden];
        delta1[i] = new double [nHidden];
    }
    
    out1 = new double [nInput];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    delta2 = new double* [nHidden];
    for (int i = 0; i < nHidden; ++i) {
        delta2[i] = new double [nInput];
    }
    
    in2 = new double [nHidden];
    out2 = new double [nHidden];
    theta2 = new double [nHidden];

	// Layer 3 - Output layer
    in3 = new double [nInput];
    out3 = new double [nInput];
    theta3 = new double [nInput];

    expected = new double [nInput];
    
    // Initialization for weights from Input layer to Hidden layer
    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; ++j) {
            int sign = rand() % 2;
            
            W[i][j] = (double)(rand() % 10 + 1) / (10.0 * nHidden);
            if (sign == 1) {
                W[i][j] = - W[i][j];
            }
        }
	}
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

void perceptron() {
    for (int i = 0; i < nHidden; ++i) {
		in2[i] = 0.0;
	}

    for (int i = 0; i < nInput; ++i) {
		in3[i] = 0.0;
	}

    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; ++j) {
            in2[j] += out1[i] * W[i][j];
		}
	}

    for (int i = 0; i < nHidden; ++i) {
		out2[i] = sigmoid(in2[i]);
	}

    for (int i = 0; i < nHidden; ++i) {
        for (int j = 0; j < nInput; ++j) {
            in3[j] += out2[i] * W[j][i];
		}
	}

    for (int i = 0; i < nInput; ++i) {
		out3[i] = sigmoid(in3[i]);
	}
}

// +---------------+
// | Norm L2 error |
// +---------------+

double squared_error(){
    double res = 0.0;
    for (int i = 0; i < nInput; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation() {
    double sum;

    for (int i = 0; i < nInput; ++i) {
        theta3[i] = out3[i] * (1.0 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 0; i < nHidden; ++i) {
        sum = 0.0;
        for (int j = 0; j < nInput; ++j) {
            sum += W[j][i] * theta3[j];
		}
        theta2[i] = out2[i] * (1.0 - out2[i]) * sum;
    }

    for (int i = 0; i < nHidden; ++i) {
        for (int j = 0; j < nInput; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
        }
	}

    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; j++) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
        }
	}

    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; j++) {
            W[i][j] += delta1[i][j] + delta2[j][i];
        }
    }
}

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

double learning_process() {
    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; ++j) {
			delta1[i][j] = 0.0;
		}
	}

    for (int i = 0; i < nHidden; ++i) {
        for (int j = 0; j < nInput; ++j) {
			delta2[i][j] = 0.0;
		}
	}

    double best_error = 1e9;
    for (int i = 0; i < nIterations; ++i) {
        perceptron();
        back_propagation();

        double error = squared_error();
        if (error < epsilon) {
			return error;
		}
        if (error > best_error) {
            return error;
        }
        best_error = error;
    }
    return best_error;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input() {
	// Reading image
    char number;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0.0; 
			} else {
				d[i][j] = 1.0;
			}
        }
	}
	
    if (showImages) {
    	cout << "Image:" << endl;
    	for (int j = 0; j < height; ++j) {
    		for (int i = 0; i < width; ++i) {
                if (d[i][j] > 0.0) {
                    cout << "1";
                } else {
                    cout << "0";
                }
    		}
    		cout << endl;
    	}
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int pos = i + j * width;
            out1[pos] = d[i][j];
        }
	}

    for (int i = 0; i < nInput; ++i) {
		expected[i] = out1[i];
	}
}

// +-------------------------------+
// | Evaluation on the testing set |
// +-------------------------------+

double evaluate_test() {
    image.open(testing_image_fn.c_str(), ios::in | ios::binary);
    char number;
    for (int i = 0; i < 16; ++i) {
        image.read(&number, sizeof(char));
    }
    double average_error = 0.0;
    for (int sample = 0; sample < nTesting; ++sample) {
        input();
        perceptron();
        average_error += squared_error() / nTesting;
    }
    image.close();
    return average_error;
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
    for (int i = 0; i < nInput; ++i) {
        for (int j = 0; j < nHidden; ++j) {
			file << W[i][j] << " ";
		}
		file << endl;
    }
	file.close();
}

// +-------------------+
// | Save color images |
// +-------------------+

void save_color_images() {
    char number;

    image.open(training_image_fn.c_str(), ios::in | ios::binary);
    ofstream output1(text_training_image_fn.c_str(), ios::out);
    for (int i = 0; i < 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int sample = 0; sample < nTraining; ++sample) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                image.read(&number, sizeof(char));
                double value = double(abs(int(number))) / 255.0;
                output1 << value << " ";
            }
        }
        output1 << endl;
    }
    output1.close();
    image.close();

    image.open(testing_image_fn.c_str(), ios::in | ios::binary);
    ofstream output2(text_testing_image_fn.c_str(), ios::out);
    for (int i = 0; i < 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int sample = 0; sample < nTesting; ++sample) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                image.read(&number, sizeof(char));
                double value = double(abs(int(number))) / 255.0;
                output2 << value << " ";
            }
        }
        output2 << endl;
    }
    output2.close();
    image.close();
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
    string line;
    while (true) {
        cout << "Do you want to see the images and the corresponding reconstructed images? (yes/no): ";
        getline(cin, line);
        if (line == "yes") {
            showImages = true;
            break;
        } 
        if (line == "no") {
            showImages = false;
            break;
        }
    }

    // Neural Network Initialization
    init_array();

    // save_color_images();

    double best_error = 1e9;
    for (int epoch = 0; epoch < Epochs; ++epoch) {
        cout << "Training epoch " << (epoch + 1) << " / " << Epochs << endl;

        image.open(training_image_fn.c_str(), ios::in | ios::binary);
        char number;
        for (int i = 0; i < 16; ++i) {
            image.read(&number, sizeof(char));
    	}
    
        for (int sample = 0; sample < nTraining; ++sample) {
            if (showImages) {
                cout << "Epoch " << (epoch + 1) << " - Sample " << (sample + 1) << endl; 
            }

            input();
    		double error = learning_process();
            
            if (showImages) {
                perceptron();
                cout << "Reconstructed image:" << endl;
                int nBits = 0;
                for (int j = 0; j < height; ++j) {
                    for (int i = 0; i < width; ++i) {
                        int pos = i + j * width;
                        if (out3[pos] > 0.5) {
                            cout << "1";
                            if (out1[pos] == 0.0) {
                                ++nBits;
                            }
                        } else {
                            cout << "0";
                            if (out1[pos] > 0.0) {
                                ++nBits;
                            }
                        }
                    }
                    cout << endl;
                }
                cout << "Reconstruction error = " << squared_error() << endl;
                cout << "Number of different pixels = " << nBits << endl;
            } 

            if ((sample + 1) % 1000 == 0) {
                cout << "    Completed training " << (sample + 1) << " examples" << endl;
            }
        }

        image.close();

        cout << "Testing epoch " << (epoch + 1) << " / " << Epochs << endl;
        double test_error = evaluate_test();
        cout << "Testing error = " << test_error << endl;
        if (test_error > best_error) {
            cout << "Early stopping!" << endl;
            break;
        }
        best_error = test_error;
	
        // Save the final network
        write_matrix(model_fn);
    }    
    return 0;
}

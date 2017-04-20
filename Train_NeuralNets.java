// Software: Training/Testing Neural Network (1 hidden layer) on MNIST dataset
// Author: Hy Truong Son
// Major: PhD. Computer Science
// Institution: Department of Computer Science, The University of Chicago
// Email: hytruongson@uchicago.edu
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2017 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.util.Scanner;
import java.util.ArrayList;
import MyLib.MLP;

public class Train_NeuralNets {
	
	// +----------+
	// | Datasets |
	// +----------+

	static String TrainingImageFileName = "MNIST/train-images.idx3-ubyte";
	static String TrainingLabelFileName = "MNIST/train-labels.idx1-ubyte";
	static String TestingImageFileName = "MNIST/t10k-images.idx3-ubyte";
	static String TestingLabelFileName = "MNIST/t10k-labels.idx1-ubyte";

	static String TextTrainingImageFileName = "saves/train-images.dat";
	static String TextTrainingLabelFileName = "saves/train-labels.dat";
	static String TextTestingImageFileName = "saves/test-images.dat";
	static String TextTestingLabelFileName = "saves/test-labels.dat";

	// +-----------------------+
	// | Files to save results |
	// +-----------------------+

	static ArrayList<String> ModelFileNames;
	static String ReportFileName;
	
	// +--------------------+
	// | Constants of model |
	// +--------------------+

	static int widthImage = 28;
	static int heightImage = 28;
	
	static int nInput = widthImage * heightImage;
	static int nHidden;
	static int nOutput = 10;
	
	// +----------------------------------------+
	// | MultiLayer Perceptron - Neural Network |
	// +----------------------------------------+

	static MLP myNet;
	
	// +---------------------+
	// | Training parameters |
	// +---------------------+

	static int Epochs;
	static double LearningRate;
	static double Momentum = 0.9;      // Momentum for stochastic learning process
	
	// +--------------------------------------+
	// | Data structures to save the datasets |
	// +--------------------------------------+

	public static class Sample {
		public double image[];
		public int label;
	}

	static ArrayList<Sample> trainData;
	static ArrayList<Sample> testData;
	
	static boolean printOutput;
	static ArrayList<Integer> performance;
	
	// +---------------------------------+
	// | Input the options from the user |
	// +---------------------------------+

	private static void inputParameters() {
		Scanner scanner = new Scanner(System.in);

		nHidden = 0;
		while (true) {
			System.out.print("Number of hidden neurons (for example, 128): ");
			nHidden = scanner.nextInt();
			if (nHidden <= 0) {
				System.out.println("The number of hidden neurons must be greater than 0");
			} else {
				break;
			}
		}

		Epochs = 0;
		while (true) {
			System.out.print("Epochs that is the number of times training the dataset (for example, 10): ");
			Epochs = scanner.nextInt();
			if (Epochs <= 0) {
				System.out.println("The number of epochs must be greater than 0");
			} else {
				break;
			}
		}

		LearningRate = 0.0;
		while (true) {
			System.out.print("Learning rate (for example, 0.01): ");
			LearningRate = scanner.nextDouble();
			if (LearningRate <= 0.0) {
				System.out.print("The learning rate must be greater than 0");
			} else {
				break;
			}
		}
		scanner.nextLine();
		
		String ModelFileName = "saves/model-nHidden-" + Integer.toString(nHidden) + "-Epochs-" + Integer.toString(Epochs) + "-LearningRate-" + Double.toString(LearningRate);
		ModelFileNames = new ArrayList<>();
		ModelFileNames.add(ModelFileName + "-Layer-0.dat");
		ModelFileNames.add(ModelFileName + "-Layer-1.dat");

		System.out.println();
		System.out.println("Model file name: " + ModelFileNames.get(0) + ", " + ModelFileNames.get(1));

		ReportFileName = "saves/report-nHidden-" + Integer.toString(nHidden) + "-Epochs-" + Integer.toString(Epochs) + "-LearningRate-" + Double.toString(LearningRate) + ".dat";
		System.out.println("Report file name: " + ReportFileName);
		System.out.println();

		printOutput = false;
		while (true) {
			System.out.print("Do you want to see the training for each example (yes/no)? ");
			String answer = scanner.nextLine();
			if (answer.equals("yes")) {
				printOutput = true;
				break;
			}
			if (answer.equals("no")) {
				printOutput = false;
				break;
			}
			System.out.println("Answer 'yes' or 'no' only!");
		}
	}

	// +--------------------------------------+
	// | Loading the datasets (images/labels) |
	// +--------------------------------------+

	private static ArrayList<Sample> loadData(String imageFileName, String labelFileName) throws IOException {
		ArrayList<Sample> data = new ArrayList<>();
		int nSamples = 0;

		// Image file
		System.out.println("Loading images from file '" + imageFileName + "'");

		BufferedInputStream imageFile = new BufferedInputStream(new FileInputStream(imageFileName));
		for (int i = 0; i < 16; ++i) {
		    imageFile.read();
		}
		
		while (imageFile.available() > 0) {
			++nSamples;
			Sample sample = new Sample();
			sample.image = new double [nInput];
			
			for (int i = 0; i < heightImage; ++i) {
				for (int j = 0; j < widthImage; ++j) {
					if (imageFile.read() > 0) {
						sample.image[j + i * widthImage] = 1.0;
					} else {
						sample.image[j + i * widthImage] = 0.0;
					}
				}
			}

			data.add(sample);
		}
		imageFile.close();

		// Label file
		System.out.println("Loading labels from file '" + labelFileName + "'");
		
		BufferedInputStream labelFile = new BufferedInputStream(new FileInputStream(labelFileName));
		for (int i = 0; i < 8; ++i) {
		    labelFile.read();
		}

		for (int sample = 0; sample < nSamples; ++sample) {
			data.get(sample).label = labelFile.read();
		}
		labelFile.close();

		// Data information
		System.out.println("Number of examples: " + Integer.toString(nSamples));

		/*
		for (int sample = 0; sample < nSamples; ++sample) {
			System.out.println("Sample " + Integer.toString(sample + 1) + ":");
			for (int i = 0; i < heightImage; ++i) {
				for (int j = 0; j < widthImage; ++j) {
					System.out.print((int)(data.get(sample).image[j + i * widthImage]));
				}
				System.out.println();
			}
			System.out.println("Label = " + data.get(sample).label);
		}
		*/

		return data;
	}

	// +----------------------+
	// | Saving data to files |
	// +----------------------+

	private static void saveData(ArrayList<Sample> data, String imageFileName, String labelFileName) throws IOException {
		System.out.println("Saving images to file " + imageFileName);
		PrintWriter imageFile = new PrintWriter(new FileWriter(imageFileName));
		for (int i = 0; i < data.size(); ++i) {
			Sample sample = data.get(i);
			for (int j = 0; j < nInput; ++j) {
				imageFile.print(sample.image[j] + " ");
			}
			imageFile.println();
		}
		imageFile.close();

		System.out.println("Saving labels to file " + labelFileName);
		PrintWriter labelFile = new PrintWriter(new FileWriter(labelFileName));
		for (int i = 0; i < data.size(); ++i) {
			labelFile.println(data.get(i).label);
		}
		labelFile.close();
	}

	// +------------------------+
	// | Network initialization |
	// +------------------------+

	private static void networkInitialization() {
		myNet = new MLP(nInput, nHidden, nOutput);
		myNet.setEpochs(10);
		myNet.setLearningRate(LearningRate);
		myNet.setMomentum(Momentum);
	}

	// +------------------------------------------+
	// | Printing an image to the standard output |
	// +------------------------------------------+

	private static void printImage(Sample sample) {
		for (int i = 0; i < heightImage; ++i) {
			for (int j = 0; j < widthImage; ++j) {
				int v = i * widthImage + j + 1;
				if (sample.image[v] > 0.0) {
					System.out.print("1");
				} else {
					System.out.print("0");
				}
			}
			System.out.println();
		}
	}

	// +----------------------------+
	// | Training / Testing Process |
	// +----------------------------+

	private static void process() throws IOException {
		performance = new ArrayList<>();
		double input[] = new double [nInput];
		double output[] = new double [nOutput];

		double best_accuracy = 0.0;
		for (int epoch = 0; epoch < Epochs; ++epoch) {
			System.out.println();
			System.out.println("----------------------------------------------------------------------------------");
			System.out.println("Epoch " + Integer.toString(epoch));
			System.out.println("Training");

			for (int i = 0; i < trainData.size(); ++i) {
				Sample sample = trainData.get(i);
				if ((i + 1) % 1000 == 0) {
					System.out.println("Done training for " + Integer.toString(i + 1) + "/" + Integer.toString(trainData.size()) + " samples");
				}
			
				if (printOutput) {
					System.out.println();
					System.out.println("Training sample " + Integer.toString(i) + ":");
					printImage(sample);
					System.out.println("Label = " + sample.label);
				}

				// Input
				for (int j = 0; j < nInput; ++j) {
					input[j] = sample.image[j];
				}

				// Output
				for (int j = 0; j < nOutput; ++j) {
					output[j] = 0.0;
				}
				output[sample.label] = 1.0;

				// Neural Network Learning
				double loss = myNet.StochasticLearning(input, output);

				if (printOutput) {
					System.out.println("Squared loss = " + Double.toString(loss));
				}
			}

			System.out.println("Testing");
			int nCorrect = 0;
			int nTesting = testData.size();
			for (int i = 0; i < nTesting; ++i) {
				Sample sample = testData.get(i);
				for (int j = 0; j < nInput; ++j) {
					input[j] = sample.image[j];
				}

				// Neural network prediction
				myNet.Predict(input, output);

				int predict = 0;
				for (int j = 0; j < nOutput; ++j) {
					if (output[j] > output[predict]) {
						predict = j;
					}
				}
				
				if (predict == sample.label) {
					++nCorrect;
				}
			}

			double accuracy = (double)(nCorrect) / (double)(nTesting);
			System.out.println("Testing accuracy = " + Integer.toString(nCorrect) + "/" + Integer.toString(nTesting) + " = " + Double.toString(accuracy));
			performance.add(nCorrect);

			if (accuracy < best_accuracy) {
				System.out.println("Early stopping!");
				break;
			} else {
				best_accuracy = accuracy;
				System.out.println("Save model to file");
				myNet.writeWeights(ModelFileNames);
			}
		}
	}

	// +--------------------+
	// | Making the summary |
	// +--------------------+

	private static void summary() throws IOException {
		System.out.println();
		System.out.println("----------------------------------------------------------------------------------");
		System.out.println("Summary:");
		int nTesting = testData.size();
		PrintWriter report = new PrintWriter(new FileWriter(ReportFileName));
		for (int epoch = 0; epoch < performance.size(); ++epoch) {
			double accuracy = (double)(performance.get(epoch)) / nTesting;
			System.out.println("Epoch " + Integer.toString(epoch) + ": Accuracy = " + Integer.toString(performance.get(epoch)) + "/" + Integer.toString(nTesting) + " = " + Double.toString(accuracy));
			report.println("Epoch " + Integer.toString(epoch) + ": Accuracy = " + Integer.toString(performance.get(epoch)) + "/" + Integer.toString(nTesting) + " = " + Double.toString(accuracy));
		}
		report.close();
	}

	// +--------------+
	// | Main Program |
	// +--------------+

	public static void main(String args[]) throws IOException {	
		// Input parameters (options) from the user
	    inputParameters();

	    // Loading the training dataset
	    trainData = loadData(TrainingImageFileName, TrainingLabelFileName);

	    // Loading the testing dataset
	    testData = loadData(TestingImageFileName, TestingLabelFileName);

	    // Saving train data to files
	    saveData(trainData, TextTrainingImageFileName, TextTrainingLabelFileName);

	    // Saving test data to files
	    saveData(testData, TextTestingImageFileName, TextTestingLabelFileName);

	    // Neural network object creation
	    networkInitialization();

	    // Training / Testing process 
	    process();

	    // Making the summary
	    summary();
	}
	
}

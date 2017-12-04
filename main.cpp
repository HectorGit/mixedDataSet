// System includes
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <stdint.h>
#include <limits.h>
#include <stdexcept>

// OpenNN includes

#include "opennn/opennn.h"
#include <opennn/multilayer_perceptron.h>
#include <opennn/perceptron_layer.h>
#include "training_strategy.h"
//#include "tests/*.h"
#include "variables.h"
//#include "statistics.h"
using namespace OpenNN;
using namespace std;

//STEP 2 - CLEAN AND ADD .DAT FILE

//STEP 3 - COPY .XML FILE TO REALTIMECLASSIFICATION PROJECT 

int main(void)
{
	try
	{
		std::cout << "OpenNN. MIXED DATASET" << std::endl;

		srand((unsigned int)time(NULL));

		/*DATA SET*/
		DataSet data_set;

		data_set.set_data_file_name("data/data.dat");

		data_set.set_separator("Comma");

		cout << "about to load data" << endl;

		data_set.load_data();

		cout << "loaded data" << endl;

		/*VARIABLES */

		Variables* variables_pointer = data_set.get_variables_pointer();

		variables_pointer->set_name(0, "Mean_Acc10_ZeroCrossings_ real ");
		variables_pointer->set_use(0, Variables::Input);

		variables_pointer->set_name(1, "Mean_Acc10_Rms_ real ");
		variables_pointer->set_use(1, Variables::Input);

		variables_pointer->set_name(2, "Mean_Acc10_Centroid_Power_powerFFT_WinHamming_ real ");
		variables_pointer->set_use(2, Variables::Input);

		variables_pointer->set_name(3, "Mean_Acc10_Rolloff_Power_powerFFT_WinHamming_ real");
		variables_pointer->set_use(3, Variables::Input);

		variables_pointer->set_name(4, "Mean_Acc10_MFCC0_Power_powerFFT_WinHamming_ real ");
		variables_pointer->set_use(4, Variables::Input);

		variables_pointer->set_name(5, "Mean_Acc10_MFCC1_Power_powerFFT_WinHamming_ real  ");
		variables_pointer->set_use(5, Variables::Input);

		variables_pointer->set_name(6, "Mean_Acc10_MFCC2_Power_powerFFT_WinHamming_ real  ");
		variables_pointer->set_use(6, Variables::Input);

		variables_pointer->set_name(7, "Mean_Acc10_MFCC3_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(7, Variables::Input);

		variables_pointer->set_name(8, "Mean_Acc10_MFCC4_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(8, Variables::Input);

		variables_pointer->set_name(9, "Mean_Acc10_MFCC5_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(9, Variables::Input);

		variables_pointer->set_name(10, "Mean_Acc10_MFCC6_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(10, Variables::Input);

		variables_pointer->set_name(11, "Mean_Acc10_MFCC7_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(11, Variables::Input);

		variables_pointer->set_name(12, "Mean_Acc10_MFCC8_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(12, Variables::Input);

		variables_pointer->set_name(13, "Mean_Acc10_MFCC9_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(13, Variables::Input);

		variables_pointer->set_name(14, "Mean_Acc10_MFCC10_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(14, Variables::Input);

		variables_pointer->set_name(15, "Mean_Acc10_MFCC11_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(15, Variables::Input);

		variables_pointer->set_name(16, "Mean_Acc10_MFCC12_Power_powerFFT_WinHamming_  real  ");
		variables_pointer->set_use(16, Variables::Input);

		variables_pointer->set_name(17, "Mean_Acc10_Kurtosis  real  ");
		variables_pointer->set_use(17, Variables::Input);

		variables_pointer->set_name(18, "Mean_Acc10_Skewness  real  ");
		variables_pointer->set_use(18, Variables::Input);

		variables_pointer->set_name(19, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_1  real  ");
		variables_pointer->set_use(19, Variables::Input);

		variables_pointer->set_name(20, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_2  real  ");
		variables_pointer->set_use(20, Variables::Input);

		variables_pointer->set_name(21, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_3  real  ");
		variables_pointer->set_use(21, Variables::Input);

		variables_pointer->set_name(22, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_4  real  ");
		variables_pointer->set_use(22, Variables::Input);

		variables_pointer->set_name(23, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_5  real  ");
		variables_pointer->set_use(23, Variables::Input);

		variables_pointer->set_name(24, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_6  real  ");
		variables_pointer->set_use(24, Variables::Input);

		variables_pointer->set_name(25, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_7  real  ");
		variables_pointer->set_use(25, Variables::Input);

		variables_pointer->set_name(26, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_8  real  ");
		variables_pointer->set_use(26, Variables::Input);

		variables_pointer->set_name(27, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_9  real  ");
		variables_pointer->set_use(27, Variables::Input);

		variables_pointer->set_name(28, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_10  real  ");
		variables_pointer->set_use(28, Variables::Input);

		variables_pointer->set_name(29, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_11  real  ");
		variables_pointer->set_use(29, Variables::Input);

		variables_pointer->set_name(30, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_12  real  ");
		variables_pointer->set_use(30, Variables::Input);

		variables_pointer->set_name(31, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_13  real  ");
		variables_pointer->set_use(31, Variables::Input);

		variables_pointer->set_name(32, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_14  real  ");
		variables_pointer->set_use(32, Variables::Input);

		variables_pointer->set_name(33, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_15  real  ");
		variables_pointer->set_use(33, Variables::Input);

		variables_pointer->set_name(34, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_16  real  ");
		variables_pointer->set_use(34, Variables::Input);

		variables_pointer->set_name(35, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_17  real  ");
		variables_pointer->set_use(35, Variables::Input);

		variables_pointer->set_name(36, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_18  real  ");
		variables_pointer->set_use(36, Variables::Input);

		variables_pointer->set_name(37, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_19  real  ");
		variables_pointer->set_use(37, Variables::Input);

		variables_pointer->set_name(38, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_20  real  ");
		variables_pointer->set_use(38, Variables::Input);

		variables_pointer->set_name(39, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_21  real  ");
		variables_pointer->set_use(39, Variables::Input);

		variables_pointer->set_name(40, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_22  real  ");
		variables_pointer->set_use(40, Variables::Input);

		variables_pointer->set_name(41, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_23  real  ");
		variables_pointer->set_use(41, Variables::Input);

		variables_pointer->set_name(42, "Mean_Acc10_SFM_Power_powerFFT_WinHamming_24  real  ");
		variables_pointer->set_use(42, Variables::Input);

		variables_pointer->set_name(43, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_1  real  ");
		variables_pointer->set_use(43, Variables::Input);

		variables_pointer->set_name(44, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_2  real  ");
		variables_pointer->set_use(44, Variables::Input);

		variables_pointer->set_name(45, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_3  real  ");
		variables_pointer->set_use(45, Variables::Input);

		variables_pointer->set_name(46, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_4  real  ");
		variables_pointer->set_use(46, Variables::Input);

		variables_pointer->set_name(47, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_5  real  ");
		variables_pointer->set_use(47, Variables::Input);

		variables_pointer->set_name(48, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_6  real  ");
		variables_pointer->set_use(48, Variables::Input);

		variables_pointer->set_name(49, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_7  real  ");
		variables_pointer->set_use(49, Variables::Input);

		variables_pointer->set_name(50, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_8  real  ");
		variables_pointer->set_use(50, Variables::Input);

		variables_pointer->set_name(51, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_9  real  ");
		variables_pointer->set_use(51, Variables::Input);

		variables_pointer->set_name(52, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_10  real  ");
		variables_pointer->set_use(52, Variables::Input);

		variables_pointer->set_name(53, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_11  real  ");
		variables_pointer->set_use(53, Variables::Input);

		variables_pointer->set_name(54, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_12  real  ");
		variables_pointer->set_use(54, Variables::Input);

		variables_pointer->set_name(55, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_13  real  ");
		variables_pointer->set_use(55, Variables::Input);

		variables_pointer->set_name(56, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_14  real  ");
		variables_pointer->set_use(56, Variables::Input);

		variables_pointer->set_name(57, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_15  real  ");
		variables_pointer->set_use(57, Variables::Input);

		variables_pointer->set_name(58, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_16  real  ");
		variables_pointer->set_use(58, Variables::Input);

		variables_pointer->set_name(59, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_17  real  ");
		variables_pointer->set_use(59, Variables::Input);

		variables_pointer->set_name(60, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_18  real  ");
		variables_pointer->set_use(60, Variables::Input);

		variables_pointer->set_name(61, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_19  real  ");
		variables_pointer->set_use(61, Variables::Input);

		variables_pointer->set_name(62, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_20  real  ");
		variables_pointer->set_use(62, Variables::Input);

		variables_pointer->set_name(63, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_21  real  ");
		variables_pointer->set_use(63, Variables::Input);

		variables_pointer->set_name(64, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_22  real  ");
		variables_pointer->set_use(64, Variables::Input);
		
		variables_pointer->set_name(65, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_23  real  ");
		variables_pointer->set_use(65, Variables::Input);

		variables_pointer->set_name(66, "Mean_Acc10_SCF_Power_powerFFT_WinHamming_24  real  ");
		variables_pointer->set_use(66, Variables::Input);

		variables_pointer->set_name(67, "Center");
		variables_pointer->set_use(67, Variables::Target);

		variables_pointer->set_name(68, "Halfedge");
		variables_pointer->set_use(68, Variables::Target);

		variables_pointer->set_name(69, "Rimshot");
		variables_pointer->set_use(69, Variables::Target);

		const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
		const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

		/*SCALING THE INPUT DATA FOR BETTER PERFORMANCE*/
		Instances* instances_pointer = data_set.get_instances_pointer();

		instances_pointer->split_random_indices();

		const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

		/*SCALING THE INPUT DATA FOR APPROXIMATION PROBLEMS*/
		const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

		/*OBTAIN AND SET INFO ABOUT THE VARIABLES OF THE DATA*/

		const size_t inputs_number = variables_pointer->count_inputs_number();//should give 67
		cout << "\n inputs number : " << inputs_number << endl;
		const size_t outputs_number = variables_pointer->count_targets_number(); //should give 3
		cout << "\n outputs number: " << outputs_number << endl;
		/*----------------------------------*/
		NeuralNetwork neural_network(inputs_number, 40, outputs_number);

		neural_network.construct_scaling_layer();

		ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

		scaling_layer_pointer->set_statistics(inputs_statistics);

		scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

		/*PROBABILISTIC LAYER - FOR PATTERN RECOGNITION & CLASSIFICATION NN'S*/
		neural_network.construct_probabilistic_layer();

		ProbabilisticLayer* probability_layer_pointer = neural_network.get_probabilistic_layer_pointer();

		probability_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);

		/*ERROR*/
		LossIndex loss_index(&neural_network, &data_set);//set both pointers at same time!
		loss_index.set_error_type("NORMALIZED_SQUARED_ERROR"); //not in the example.
		loss_index.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

		/*TRAINING STRATEGY*/
		TrainingStrategy training_strategy(&loss_index);
		training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);
		QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
		quasi_Newton_method_pointer->set_maximum_iterations_number(60); //changed this AGAIN
		quasi_Newton_method_pointer->set_display_period(10);
		training_strategy.perform_training();

		TestingAnalysis testing_analysis(&neural_network, &data_set);
		Matrix<size_t> confusion = testing_analysis.calculate_confusion();

		scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);

		neural_network.save_expression("expresion.txt");
		data_set.save("data/proceeds/data_set.xml");

		neural_network.save("data/proceeds/neural_network.xml");
		neural_network.save_expression("data/proceeds/expression.txt");
		confusion.save("data/proceeds/confusion.dat");

		training_strategy.save("data/proceeds/training_strategy.xml");

		return(0);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;

		return(1);
	}

}
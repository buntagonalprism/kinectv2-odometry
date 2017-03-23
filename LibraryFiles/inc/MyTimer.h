/***************************************************************************************

File Name:		MyTimer.h
Author:			Alex Bunting
Date Modified:  21/7/14

Description:
Simple class implementing microsecond precision timing capabilities using the Boost::chrono
library. Used for benchmarking program performance with the tic-toc Matlab style functions
and for recording timestamps of data observations. 

****************************************************************************************/


#ifndef MY_TIMER
#define MY_TIMER

#include <iostream>		
#include <boost/chrono/chrono.hpp>

using namespace boost::chrono;
using namespace std;

class Timer {
private:
	high_resolution_clock::time_point time1;
	high_resolution_clock::time_point time2;
	high_resolution_clock::time_point time_now;
	
public:
	// Saves current clock time
	void tic(void) {
		time1 = high_resolution_clock::now();
	}

	// Prints the time since the last 'tic()' call
	void toc(void) {
		time2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<microseconds> (time2 - time1);
		cout << "Time elapsed: " << time_span.count() << " seconds." << endl;
	}

	// Prints the time with an additional message
	void toc(char* message) {
		time2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<microseconds> (time2 - time1);
		cout << message << ": Time elapsed: " << time_span.count() << " seconds." << endl;
	}

	// Overload takes bool to save or not
	// Intended to allow single-variable switching between
	// timed and non-timed operation in main code
	void tic(bool save) {
		if (save)
			this->tic();
	}

	// Overload takes single bool to print elapsed time or not
	void toc(bool print) {
		if (print)
			this->toc();
	}

	// Overload takes bool to print or not, with additional message if so
	void toc(char* message, bool print) {
		if (print)
			this->toc(message);
	}

	

	void saveTime(void) {
		time_now = high_resolution_clock::now();
	}
	
	microseconds printTime(void) {
		return duration_cast<microseconds>(time_now.time_since_epoch());
	}
	
};

#endif
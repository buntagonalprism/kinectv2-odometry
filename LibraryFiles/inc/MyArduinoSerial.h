/***************************************************************************************

File Name:		MyArduinoSerial.h
Author:			Alex Bunting
Date Modified:  21/7/14

Description:
Contains the MyArudinoSerial class, used to connect to an Arudino device using USB serial
and recieve data for it. Includes a function for synchronising the time between the 
Arduino and PC which should be called for sync data capture. Requires the correct program
to be running on the Arduino - currently SyncIMUGPS.ino

TODO
 - Add a timeout to the SyncTime function to identify incorrect Baud rate or wrong program
     running on the Arduino

****************************************************************************************/

#ifndef MY_ARDUINO_SERIAL
#define MY_ARDUINO_SERIAL

#define ARDUINO_WAIT_TIME 2000

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/chrono/chrono.hpp>

class MyArduinoSerial
{
public:
	// Initialize Serial communication with the given COM port
	// [in]  The name of the COM port to open
	// [in]  The BAUD rate to communicate with
	MyArduinoSerial(LPCWSTR portName, DWORD baudRate);


	// Close the connection
	// NOTE: for some reason you can't connect again before exiting
	// the program and running it again
	~MyArduinoSerial();


	// Read data in a buffer, 
	// [out] The buffer to read in to
	// [in]  The number of bytes to read 
	//       If the number to read is greater than available data, only the 
	//       available data will be returned
	// [ret] The number of bytes actually read, or -1 on read failure / no data
	int ReadData(char *buffer, unsigned int nbChar);


	// Reads data until a newline or null terminator is reached
	// This is blocking, and should only be used in testing situations
	// [in]  Buffer to read in to
	// [in]  Size of the buffer
	// [out] Number of bytes read in the line	
	int ReadLine(char *buffer, unsigned int nbChar);

	// Writes data from a buffer through the Serial connection
	// [in]  The data buffer to write
	// [in]  The number of bytes to write from that buffer
	// [ret] True on success
	bool WriteData(char *buffer, unsigned int nbChar);


	//Check if we are actually connected
	bool IsConnected();


	// Checks for a valid arduino connection then synchronises Arduino time with PC
	void SyncConnection();



private:
    //Serial comm handler
    HANDLE hMyArduinoSerial;

    //Connection status
    bool connected;

    //Get various information about the connection
    COMSTAT status;

    //Keep track of last error
    DWORD errors;

	// Arudino-PC time synchronisation function
	// Requests a reset of the Arduino, waits for Sync request enquirey
	// Requests the current time from the arduino, and finds an offset to PC time
	// Transmits this offset to Arudino and waits for acknowledgement
	void SyncTime(void);

};

#endif MY_ARDUINO_SERIAL
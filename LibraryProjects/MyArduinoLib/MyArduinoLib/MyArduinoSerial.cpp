#include "MyArduinoSerial.h"

MyArduinoSerial::MyArduinoSerial(LPCWSTR portName, DWORD baudRate)
{
    //We're not yet connected
    this->connected = false;

    //Try to connect to the given port throuh CreateFile
    this->hMyArduinoSerial = CreateFile(portName,
            GENERIC_READ | GENERIC_WRITE,
            0,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

    //Check if the connection was successfull
    if(this->hMyArduinoSerial==INVALID_HANDLE_VALUE)
    {
        //If not success full display an Error
        if(GetLastError()==ERROR_FILE_NOT_FOUND){

            //Print Error if neccessary
            printf("ERROR: Handle was not attached. Reason: %s not available.\n", portName);

        }
        else
        {
            printf("ERROR!!!");
        }
    }
    else
    {
        //If connected we try to set the comm parameters
        DCB dcbSerialParams = {0};

        //Try to get the current
        if (!GetCommState(this->hMyArduinoSerial, &dcbSerialParams))
        {
            //If impossible, show an error
            printf("failed to get current serial parameters!");
        }
        else
        {
            //Define serial connection parameters for the arduino board
            dcbSerialParams.BaudRate=baudRate;
            dcbSerialParams.ByteSize=8;
            dcbSerialParams.StopBits=ONESTOPBIT;
            dcbSerialParams.Parity=NOPARITY;

             //Set the parameters and check for their proper application
             if(!SetCommState(hMyArduinoSerial, &dcbSerialParams))
             {
                printf("ALERT: Could not set Serial Port parameters");
             }
             else
             {
                 //If everything went fine we're connected
                 this->connected = true;
                 //We wait 2s as the arduino board will be reseting
                 Sleep(ARDUINO_WAIT_TIME);
             }
        }
    }

}

MyArduinoSerial::~MyArduinoSerial()
{
    //Check if we are connected before trying to disconnect
    if(this->connected)
    {
        //We're no longer connected
        this->connected = false;
        //Close the serial handler
        CloseHandle(this->hMyArduinoSerial);
    }
}

int MyArduinoSerial::ReadData(char *buffer, unsigned int nbChar)
{
    //Number of bytes we'll have read
    DWORD bytesRead;
    //Number of bytes we'll really ask to read
    unsigned int toRead;

    //Use the ClearCommError function to get status info on the Serial port
    ClearCommError(this->hMyArduinoSerial, &this->errors, &this->status);

    //Check if there is something to read
    if(this->status.cbInQue>0)
    {
        //If there is we check if there is enough data to read the required number
        //of characters, if not we'll read only the available characters to prevent
        //locking of the application.
        if(this->status.cbInQue>nbChar)
        {
            toRead = nbChar;
        }
        else
        {
            toRead = this->status.cbInQue;
        }

        //Try to read the require number of chars, and return the number of read bytes on success
        if(ReadFile(this->hMyArduinoSerial, buffer, toRead, &bytesRead, NULL) && bytesRead != 0)
        {
            return bytesRead;
        }

    }

    //If nothing has been read, or that an error was detected return -1
    return -1;

}


int MyArduinoSerial::ReadLine(char *buffer, unsigned int nbChar)
{
	//Number of bytes we'll have read
	DWORD bytesRead = 0, dummy;
	//Number of bytes we'll really ask to read
	unsigned int toRead;
	buffer[bytesRead] = 'D'; // For dummy

	// Poll for a full line of data
	do  {
		if (this->status.cbInQue>0)
			ReadFile(this->hMyArduinoSerial, &buffer[bytesRead], 1, &dummy, NULL);
	} while ((bytesRead < nbChar - 1 && buffer[bytesRead] != '\n' && buffer[bytesRead++] != '\0'));
	buffer[bytesRead] = '\0';

	//If nothing has been read, or that an error was detected return -1
	return bytesRead;

}

bool MyArduinoSerial::WriteData(char *buffer, unsigned int nbChar)
{
    DWORD bytesSend;

    //Try to write the buffer on the Serial port
    if(!WriteFile(this->hMyArduinoSerial, (void *)buffer, nbChar, &bytesSend, 0))
    {
        //In case it don't work get comm error and return false
        ClearCommError(this->hMyArduinoSerial, &this->errors, &this->status);

        return false;
    }
    else
        return true;
}

bool MyArduinoSerial::IsConnected()
{
    //Simply return the connection status
    return this->connected;
}

void MyArduinoSerial::SyncConnection() {
	if (this->IsConnected()) {
		printf("Successfully connected to Arduino\n");
		this->SyncTime();
	}
	else {
		printf("Connection to arduino failed - try checking the COM port number\n");
		printf("Press any key to exit\n");
		std::cin.ignore();
		exit(EXIT_FAILURE);
	}
}

void MyArduinoSerial::SyncTime(void) {
	
	std::cout << "Synchronising time with Arduino: Calling reset and getting Arduino time..." << std::endl;

	// Command a soft reset, pause until sycn enquiry receivsed
	this->WriteData("R",1);
	char ack[2]="\0";
	Sleep(10);
	while( ack[0] !=  0x05) {this->ReadData(ack,1);}
		

	// Save the current time and request arduino time
	boost::chrono::high_resolution_clock::time_point now = boost::chrono::high_resolution_clock::now();
	this->WriteData("S",1);
	char timeString[255] = "\0";
	int dataLength = 255;
	//Sleep(10);
	while (this->ReadData(timeString,dataLength) == -1) {}
	std::cout << "Received Arudino Time: " << timeString << std::endl;
	long arduinoTime;
	sscanf(timeString,"%d",&arduinoTime);

	// Get the offset between the arduino time and the PC time in milliseconds and send it back
	boost::chrono::milliseconds now_milli = boost::chrono::duration_cast<boost::chrono::milliseconds>(now.time_since_epoch());
	boost::chrono::microseconds now_micro = boost::chrono::duration_cast<boost::chrono::microseconds>(now.time_since_epoch());
	boost::chrono::milliseconds offset = now_milli - boost::chrono::milliseconds(arduinoTime); 
	char returnTime[255] = {0};
	sprintf(returnTime,"%d",offset);
	std::cout << "Time offset is: " << returnTime << std::endl;
	this->WriteData(returnTime,255);

	// Wait for acknowledgement
	while( ack[0] !=  0x06) {this->ReadData(ack,1);}
	return;
	
}
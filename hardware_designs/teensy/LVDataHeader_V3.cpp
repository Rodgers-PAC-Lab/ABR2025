// 1/08/17  WNG
// See Google doc "Teensy header for serial data to Labview".
//#define ADC_BUFFER_SIZE_SAMPLES  (RF_PAYLOAD_SAMPLES * RF_BUF_PER_ADC_BUF + RF_PAYLOAD_SAMPLES_LAST)
//#define ADC_BUFFER_SIZE_BYTES    (ADC_BUFFER_SIZE_SAMPLES * 2)  //in bytes

// Alternate method of syncing using start and end bytes, and escaped values:
// http://eli.thegreenplace.net/2009/08/12/framing-in-serial-communications
//
// Possible improvement to my current scheme:  Add 2-byte checksum at end of data.


// 10/22/18 - *** To Do:  LVCmdAvail should check for string overflow and return error.


#include "LVDataHeader_V3.h"
#include "Arduino.h"

#define SERIAL_DEBUG true


String pkt_name(PKT_TYPES p) {
// Returns String name of the packet type.
	if (p < DEFINED_PKT_TYPES) {
		return (PKT_TYPE_NAME[p]);
	} else {
		return (String("INVALID PKT TYPE: ") + String(p));
	}
}


void writeU8(uint8_t t, uint8_t &checksum) {
	Serial.write(t);
	checksum += t;
}

	
void writeU16(uint16_t t, uint8_t &checksum) {
	uint8_t tmp = uint8_t(t / 256);
	Serial.write(tmp); //msb first allows LV to type cast to I16
	checksum += tmp;
	tmp = uint8_t(t % 256);
	Serial.write(tmp); //lsb
	checksum += tmp;
}

/* Labview states for Type 1A header
0 AA
1 55
2 5A
3 A5
4 pkt type
5 hdr size
6 datarep
7 npktsseq
8 seqnum
9 dataclass
10 nsamp
11 samprate
12 auxSamples (was fftpoints)
13 nchan
14 checksum
15 ok
16 error
*/


void writeLVStruct(PKT_TYPES pkt_type, io_struct_t ios) {
//Write command-return header and data to Labview, such as Query specs struct.
//This is unpacked in Labview in Get General Header.vi, "Reform Header" case.
//*** Each structure must conform to a LV typedef control, in both size of each field
//    and order of the fields within the struct.  E.g., PKT_LOG_INFO must match
//    "Log info message type typedef.ctl".
//*** Must always write packet_type, header_size, dataRep, and command, in that order,
//then body data, then checksum.
//header_size: start with datarep and go to (including) checksum.

	uint8_t HEADER_SIZE; 
	uint8_t datarep = 1;

	Serial.write(0xAA); // 4 sync bytes
	Serial.write(0x55);
	Serial.write(0x5A);
	Serial.write(0xA5);
	uint8_t checksum = 0;
	writeU8(pkt_type, checksum); //pkt_type is always the 1st item to write

	switch (pkt_type) {
		case PKT_QUERY_SPECS:
		case PKT_RECV_PRESENT: //receiver present; used in init process while polling serial ports
			HEADER_SIZE = 19; // datarep + command + 8*2 + checksum = 19 bytes
			writeU8(HEADER_SIZE,	 checksum);  //everything AFTER this field counts in header_size
			writeU8(datarep,		 checksum);
			writeU8(0,						checksum);
			writeU16(ios.qs.device_id,		checksum);
			writeU16(ios.qs.board_version,	checksum);
			writeU16(ios.qs.nchan_max,		checksum);
			writeU16(ios.qs.sample_rate_max, checksum);
			writeU16(ios.qs.bit_depth,		checksum);
			writeU16(ios.qs.pos_FS,			checksum);
			writeU16(ios.qs.neg_FS,			checksum);
			writeU16(ios.qs.zero_offset,	 checksum);
			Serial.write(checksum);
			break;

		case PKT_LOG_INFO:
			// Must conform to LV "Log info message type typedef.ctl".
			// The first field, Loginfo msg type, is a U16.19
			HEADER_SIZE = 17; 
			writeU8(HEADER_SIZE, checksum);  //everything AFTER this field counts in header_size
			writeU8(datarep,     checksum);
			writeU8(0,           checksum);
			writeU16(ios.li.info_msg_t,    checksum); //defined as U16 in LV
			writeU16(uint16_t(ios.li.lv_time >> 16),    checksum); //hi word
			writeU16(uint16_t(ios.li.lv_time & 0xFFFF), checksum); //lo word
			writeU16(ios.li.pkt_index,     checksum);
			writeU16(ios.li.tries,         checksum);
			writeU16(ios.li.prog_location, checksum);
			writeU16(ios.li.pkt_seq,       checksum);
			Serial.write(checksum);
			break;

		default: //for simple msg ack, PKT_LV_DATA, etc, that use "sp" struct
			HEADER_SIZE = 12;	// datarep to checksum (inclusive) = 12 bytes.
								// does not include HEADER_SIZE.
			writeU8(HEADER_SIZE,        checksum); 
			writeU8(ios.sp.dataRep,     checksum);
			writeU8(ios.sp.nPktsSeq,    checksum);
			writeU8(ios.sp.seqNum,      checksum);
			writeU8(ios.sp.dataClass,   checksum);
			writeU16(ios.sp.nSamp,      checksum);
			writeU16(ios.sp.sampRate,   checksum);
			writeU16(ios.sp.trig_in_sampleCount, checksum);
			writeU8(ios.sp.nChan, checksum);
			Serial.write(checksum);
			break;

	} //end switch
} // end writeLVStruct


void sendLVData16(uint16_t buf[], int nChan, int nSamp, int total_samples, int sampRate, uint8_t seqNum) {
// Write header and one packet-set of data to Labview.
// nSamp is # samples per channel (aka # observations), e.g. 16.
// buf is a 1D-array, size total_samples of uint16_t.
// Assuming SCHEME_15_4, this sends 4 full packets of data plus 1 partial packet.
// Assuming nChan = 4 and nSamp = 16, this equals 64 data samples plus 11 words of aux data,
// so total_samples = 75.
// ADC data is nChan*nSamp, but total_samples is greater than this because of aux_data in last packet.

	int serBufSize = total_samples * 2; //bytes
	uint8_t serBuf[serBufSize]; //byte array for Serial.write
	io_struct_t ios;
	
	ios.sp.pktType    = PKT_LV_DATA;
	ios.sp.dataRep    = DATA_REP_I16;
	ios.sp.nPktsSeq   = 1;
	ios.sp.seqNum     = seqNum;
	ios.sp.dataClass  = DATA_CLASS_PT_ADC;
	ios.sp.nSamp      = nSamp;
	ios.sp.nChan      = nChan;
	ios.sp.sampRate   = sampRate;
	ios.sp.trig_in_sampleCount = -1; // -1 = no trigger
	writeLVStruct(PKT_LV_DATA, ios);
	for (int i = 0; i < total_samples; i++) {  //convert i16 to i8 in serBuf
		serBuf[i*2]   = uint8_t((buf[i] & 0xFF00) >> 8); //msb first 
		serBuf[i*2+1] = uint8_t( buf[i] & 0x00FF);       //lsb
	}
	Serial.write(serBuf, serBufSize);
} //end sendLVData16


void sendLVData32(int32_t buf[], int nChan, int nSamp, int total_samples, int sampRate, uint8_t seqNum,
				int16_t trig_in_sampleCount) {
// For ADS1299, Rodgers lab.  Samples are 32-bit.
// Write header and one packet-set of data to Labview.
// nSamp is # samples per channel (aka # observations).
// buf is a 1D-array, size total_samples of int32_t.
// ADC data size is nChan*nSamp.

	const int BYTES_PER_SAMPLE = 4;
	int serBufSize = total_samples * BYTES_PER_SAMPLE; //bytes; each sample is 4 bytes
	uint8_t serBuf[serBufSize]; //byte array for Serial.write
	io_struct_t ios;
	
	ios.sp.pktType    = PKT_LV_DATA;
	ios.sp.dataRep    = DATA_REP_I32; // new data rep code for ADS1299
	ios.sp.nPktsSeq   = 1;
	ios.sp.seqNum     = seqNum;
	ios.sp.dataClass  = DATA_CLASS_PT_ADC;
	ios.sp.nSamp      = nSamp;
	ios.sp.nChan      = nChan;
	ios.sp.sampRate   = sampRate;
	ios.sp.trig_in_sampleCount = trig_in_sampleCount; // where trigger occurs w/i packet
	writeLVStruct(PKT_LV_DATA, ios);
	for (int i = 0; i < total_samples; i++) {  //convert i32 to i8 in serBuf
		serBuf[i*BYTES_PER_SAMPLE]   = uint8_t((buf[i] & 0xFF000000) >> 24); //msb first 
		serBuf[i*BYTES_PER_SAMPLE+1] = uint8_t((buf[i] & 0x00FF0000) >> 16);
		serBuf[i*BYTES_PER_SAMPLE+2] = uint8_t((buf[i] & 0x0000FF00) >>  8);
		serBuf[i*BYTES_PER_SAMPLE+3] = uint8_t( buf[i] & 0x000000FF);       //lsb
	}
	Serial.write(serBuf, serBufSize);
} //end sendLVData32






void sendLVCommand(PKT_TYPES pkt_type, io_struct_t ios) {
// Write command packet to Labview. 
	writeLVStruct(pkt_type, ios);
} //end sendLVCommand


void tokenizeCmdString(char str[], char *cmd, long int params[], int &num_params) {
// *** Fill cmd with some chars before calling! 
// Expects 1st token to be a string, then numbers. Delimiter is comma.
// Best explanation of strtok is here:  http://www.cplusplus.com/reference/cstring/strtok/
// First call of strtok expects a string; subsequent calls expect NULL as first argument.
// *** NOTE: By convention, size of params array must be at least 10 ***
// 11/02/18 - changed atoi to atol so we can have long int params.

	const char sep[] = ",";
	char *token;
	
	token = strtok(str, sep); // get the first token 
	int index = 0;
	while( token != NULL ) {	// walk through other tokens
		//Serial1.print("	tok = ");
		//Serial1.println(token);
		//Serial1.print("	len token = ");
		//Serial1.println(strlen(token));
		if (index == 0) { //1st token is a character
			if (strlen(cmd) >= strlen(token)) {
				strcpy(cmd, token);
			}
		} else { //subsequent tokens are numbers
			params[index-1] = atol(token);
		}
		token = strtok(NULL, sep);
		index++;
	} //endwhile
	num_params = index-1;
} //end tokenize


bool LVCmdAvail(char &cmdChar, long int params[], int &N) {
  // Checks serial port from Labview.
  // Expects a single alpha character, comma, digits, comma, digits, semicolon.
  // Reads string until we get ";" termination character, e.g. "S,8,125;"
  // Params are passed by reference so we can pass values back to the caller
  // Returns true for complete command, false for no data, incomplete, or timeout.
  // *** NOTE: By convention, size of params array must be at least 10 ***

	const int strSize = 64;
	const unsigned long time_limit = 1000; //ms to wait for end of command
	char str[strSize];  
	int index = 0;
	char ch = ' '; //init to non-semicolon
	unsigned long start_time;
	boolean time_out;
	char cmd[] = "        "; //must fill w/ some chars before calling tokenize

	if (Serial.available()) {
		start_time = millis();
		time_out = false;
		while (Serial.available() && (ch != ';') && !time_out && index < (strSize-1)) {
			ch = Serial.read();
			str[index] = ch; // add the character to the string;
			index++;
			delayMicroseconds(500); // wait for next char to come in.  was 1ms
			time_out = millis() > (start_time + time_limit);
		} // endwhile
		if (time_out) {
			return false;
		} else {
			str[index] = 0; //add null char to end of string
			tokenizeCmdString(str, cmd, params, N);
			cmdChar = toupper(cmd[0]);
			if (SERIAL_DEBUG) {
				Serial1.print("LVCmdAvail: ");  Serial1.println(cmdChar);
			}
			return true;
		}
	} else {  //nothing at serial port
		return false;
	}
} // end LVCmdAvail


void tokenizeCmdString_float(char str[], char *cmd, float params[], int &num_params) {
// *** Fill cmd with some chars before calling! 
// For float params.

	const char sep[] = ",";
	char *token;
	token = strtok(str, sep); // get the first token 
	int index = 0;
	while( token != NULL ) {	// walk through other tokens
		//Serial1.print("	tok = ");
		//Serial1.println(token);
		//Serial1.print("	len token = ");
		//Serial1.println(strlen(token));
		if (index == 0) { //1st token is a character
			if (strlen(cmd) >= strlen(token)) {
				strcpy(cmd, token);
			}
		} else { //subsequent tokens are numbers
			params[index-1] = float(atof(token)); //convert to float; atof is double
		}
		token = strtok(NULL, sep);
		index++;
	} //endwhile
	num_params = index-1;
} //end tokenize float


bool LVCmdAvail_float(char &cmdChar, float params[], int &N) {
  // As LVCmdAvail() except for float params.
  // Checks serial port from Labview.
  // Expects a single alpha character, comma, digits, comma, digits, semicolon.
  // Reads string until we get ";" termination character, e.g. "S,8.0,125.3;"
  // Params are passed by reference so we can pass values back to the caller
  // Returns true for complete command, false for no data, incomplete, or timeout.
  // *** NOTE: By convention, size of params array must be at least 10 ***

	const int strSize = 128;
	const unsigned long time_limit = 1000; //ms to wait for end of command
	char str[strSize];  
	int index = 0;
	char ch = ' '; //init to non-semicolon
	unsigned long start_time;
	boolean time_out;
	char cmd[] = "        "; //must fill w/ some chars before calling tokenize

	if (Serial.available()) {
		start_time = millis();
		time_out = false;
		while (Serial.available() && (ch != ';') && !time_out && index < (strSize-1)) {
			ch = Serial.read();
			str[index] = ch; // add char to string;
			index++;
			delayMicroseconds(100); // wait for next char to come in.  was 500us
			time_out = millis() > (start_time + time_limit);
		} // endwhile
		if (time_out) {
			return false;
		} else {
			str[index] = 0; //add null char to end of string
			tokenizeCmdString_float(str, cmd, params, N);
			cmdChar = toupper(cmd[0]);
			if (SERIAL_DEBUG) {
				Serial1.print("LVCmdAvail: ");  Serial1.println(cmdChar);
			}
			return true;
		}
	} else {  //nothing at serial port
		return false;
	}
} // end LVCmdAvail











// Used by Plessey Teensy Receiver to write data packets to Labview over USB serial connection.
// 1/08/17
// *** This spec must match Google doc "Header for serial data to Labview - Plessey Teensy" ***

// 3/29/23 - Chris Rodgers, ADS1299 prototype and Teensy 4.0.
//    Radio not used; purely USB.
//    adc_specs_t field changed to "gain".
//    adc_query_specs_t pos_fs, neg_fs and zero_offset changed to int16_t.

// Plessey Acquire V2 (Heidi) - *** Set defines to choose Rodgers or Plessey Acquire
// 6/13/23
// Changed LVHdrSpecs.auxSamples field name to trig_in_sampleCount and type
//   from uint16_t to int16_t.

#include <Arduino.h>

#define LVDataHeader_V3

#define RODGERS_LAB      // either Rodgers or Plessey Acquire
#undef  PLESSEY_ACQUIRE


// Define buffer size, no longer dependent on RF packet size.
// - Let LV choose samples/loop, probably 500.  ADC_BUFFER_SIZE_SAMPLES can be
//   larger than this as long as the total bytes per transfer is less than 61440,
//   the EDF file limit for one write.  1000Hz * 24ch (w/T/H) * 2 = 48000 bytes.

// ADS1299 version        :  500*8 samples, bytes/sample = 4
// Plessey Acquire version:  500   samples, bytes/sample = 2

#ifdef RODGERS_LAB
// ADC defines for ADS1299
const int ADC_BUFFER_SIZE_SAMPLES = (500*8);	// this must be a multiple of 8 (channels)
												// so 4000 holds 8 ch. x 500 readings
const int BYTES_PER_ADC_SAMPLE    = 4; // samples are int32_t
const int ADC_BUFFER_SIZE_BYTES   = (ADC_BUFFER_SIZE_SAMPLES * BYTES_PER_ADC_SAMPLE);
#endif

#ifdef PLESSEY_ACQUIRE
// Plessey Acquire V2, Heidi
const int ADC_BUFFER_SIZE_SAMPLES = 500;
const int BYTES_PER_ADC_SAMPLE    = 2; // samples are int16_t
const int ADC_BUFFER_SIZE_BYTES   = (ADC_BUFFER_SIZE_SAMPLES * BYTES_PER_ADC_SAMPLE);
#endif



// Packet types for data and command packets.
// *** Must match LV "Comm Commands typedef.ctl" or is it "Packet types typedef.ctl".
// Max of 32 items because field is 5 bits wide in rf_data[0].
// A valid command is "((pkt_type > PKT_1ST_COMMAND) && (pkt_type < PKT_TYPE_LAST))"

// *********************************************************************************
// *** ALWAYS UPDATE THE NEXT 3 ITEMS AND LV "PACKET TYPES TYPEDEF.CTL" TOGETHER ***
// *********************************************************************************

const int DEFINED_PKT_TYPES = 25;   //one more than the last defined type in enum below

enum PKT_TYPES {PKT_NULL             =  0,
				PKT_DATA_154         =  1, PKT_DATA_154_LAST        =  2,
				PKT_DATA_154_RESEND  =  3, PKT_DATA_154_LAST_RESEND =  4,
				// 5-6 reserved for future data types
				PKT_LV_DATA       =  7,  //from receiver to LV
				PKT_1ST_COMMAND   =  8,  //first command item; previous types are data
				PKT_STREAM_START  =  9, PKT_STREAM_STOP   = 10,  PKT_MSG_ACK       = 11,
				PKT_IDLE          = 12, PKT_ERROR         = 13,  PKT_QUERY_SPECS   = 14,
				PKT_LOG_INFO      = 15, PKT_ACK_PAYLOAD   = 16,  PKT_ADC_SPECS     = 17,
				PKT_SEND_EFFECTOR = 18, PKT_READ_EFFECTOR = 19,  PKT_RECV_PRESENT  = 20,
				PKT_NOT_FOUND     = 21, PKT_SYNC_TIME     = 22,  PKT_RESEND        = 23,
				PKT_TYPE_LAST     = 24}; //move this down as we add more valid types

const String PKT_TYPE_NAME[DEFINED_PKT_TYPES] = {
"PKT_NULL",						"PKT_DATA_154",
"PKT_DATA_154_LAST",			"PKT_DATA_154_RESEND",
"PKT_DATA_154_LAST_RESEND",		"DATARES5",
"DATARES6",						"PKT_LV_DATA",
"PKT_1ST_COMMAND",				"PKT_STREAM_START",
"PKT_STREAM_STOP",				"PKT_MSG_ACK",
"PKT_IDLE",						"PKT_ERROR",
"PKT_QUERY_SPECS",				"PKT_LOG_INFO",
"PKT_ACK_PAYLOAD",				"PKT_ADC_SPECS",
"PKT_SEND_EFFECTOR",			"PKT_READ_EFFECTOR",
"PKT_RECV_PRESENT",				"PKT_NOT_FOUND",
"PKT_SYNC_TIME",				"PKT_RESEND",
"PKT_TYPE_LAST"};


// Use this function to get String name of a packet type.
// Avoid indexing into array above - this fcn does bounds checking.
String pkt_name(PKT_TYPES p);


// This goes into nSamp field in LVHdrSpecs. Must match LV "Log info message type typedef.ctl"
enum LOGINFO_MSG_T {LOG_DUP_PKT, LOG_MISSING_PKT, LOG_PKT_TIMEOUT, LOG_GENERAL_INFO, 
                    LOG_QUEUE_ERROR, LOG_WRONG_PKT_TYPE, LOG_PKT_RECOVERED}; 


struct adc_specs_t { // aka "as"
	uint16_t nchan_acquire;   // #channels to acquire with ADC
	uint16_t nchan_keep;      // #channels to keep/store after being acquired
	uint16_t sample_rate_hz;
	uint16_t packet_scheme;
	uint16_t bit_depth;       //bit depth and FS range can be used to find volts/LSB
	uint16_t sample_averages;
	uint16_t gain;            // ADC PGA gain setting
	uint16_t device_id;
	uint16_t flags;           // extra flags to send to transmitter
};

// Specs from transmitter board - specifies type of ADC on board, its capabilities,
// and basic board data.  Also used to ID receiver, but receiver only sends device_id and board_version.
struct adc_query_specs_t { // aka "qs"
	uint16_t device_id;       // unique ID for xmit & recv boards. MS byte = lab, LS = board #. Start at 1.
	uint16_t board_version;   // latest Xmit is 3
	uint16_t nchan_max;       // max #channels ADC can acquire
	uint16_t sample_rate_max; // ADC max sample rate, considering RF link capability
	uint16_t bit_depth;       // bit depth and FS range can be used to find volts/LSB
	int16_t  pos_FS;          // pos. full scale input, as mV, e.g. 5120
	int16_t  neg_FS;          // as mV (negative), e.g. -5120
	int16_t  zero_offset;     // integer to be subtracted from raw data
};


// Type 1A header from Teensy to LV. It precedes data packets.
struct LVHdrSpecs {      // aka "sp"
	uint8_t  pktType;    // 
	uint8_t  dataRep;    // 1=I16, 2=F32.  
	uint8_t  nPktsSeq;   // #pkts in entire sequence.
	uint8_t  seqNum;     // seq. # for this packet, 1-based
	uint8_t  dataClass;  // 1=raw a/d data, 2=FFT mag., 2D ADC
	uint16_t nSamp;      // #samples after hdr, in data pkts
	uint16_t sampRate;   // sample rate for raw data (Hz)
	int16_t  trig_in_sampleCount; // count w/i pkt where trigger occurred, or -1
	uint8_t  nChan;      // #channels
};

// Must match LV "Log info cluster typedef.ctl"
struct log_info_t {
	uint16_t info_msg_t;
	long int lv_time;     // time send from Xmit or Recv
	uint16_t pkt_index;
	uint16_t pkt_seq;
	uint16_t tries;
	uint16_t prog_location;
};

union io_struct_t { //aka "ios"
	adc_specs_t            as;
	adc_query_specs_t      qs;
	LVHdrSpecs             sp;
	log_info_t             li;
};


// Labview-to-Receiver Commands:
// S - start
// X - stop
// Q - query specs
// A - Ack (message)
// I - Idle
// E - Error
// O - Output (send effector)
// P - Packet Ack
// R - Read effector
// Z - Null

// *** This list of packet types must match LV "Packet types typedef.ctl"!
// Packet type has 5 bits in header so there can be up to 32 types.
const uint8_t LV_PACKET_TYPE_NULL     =  0;
const uint8_t LV_PACKET_TYPE_SPECTRUM =  1; // used in spectrum_wng.ino and FFT Plotter.vi
const uint8_t LV_PACKET_TYPE_PT_DATA  =  2; // PT = Plessey-Teensy.  multi_chan xmit and recv.
const uint8_t LV_PACKET_TYPE_CONTROL  =  3; // Commands to/from receiver. Field nPktsSeq holds the command

// These must match Labview "Data Class typedef.ctl", in teensy_xmit_recv_test.llb
const uint8_t DATA_CLASS_SPECTRUM_RAW  =  1;
const uint8_t DATA_CLASS_SPECTRUM_FFT  =  2;
const uint8_t DATA_CLASS_PT_ADC        =  3; //2D array of I16 ADC data

const uint8_t DATA_REP_I16 = 1; // Plessey
const uint8_t DATA_REP_F32 = 2;
const uint8_t DATA_REP_I32 = 3; // ADS1299


void writeLVStruct(PKT_TYPES pkt_type, io_struct_t ios);

void sendLVData16(uint16_t buf[], int nChan, int nSamp, int total_samples, int sampRate, uint8_t seqNum);
	// For Plessey programs.  Write header and one packet set of data to Labview.
	// buf is a 1D-array, size total_samples of uint16_t.
	// ADC data is nChan*nSamp.

void sendLVData32(int32_t buf[], int nChan, int nSamp, int total_samples, int sampRate, uint8_t seqNum,
				int16_t trig_in_sampleCount);
// For ADS1299, Rodgers lab.  Samples are 32-bit.
// Write header and one packet-set of data to Labview.
// nSamp is # samples per channel (aka # observations).
// buf is a 1D-array, size total_samples of int32_t.
// ADC data size is nChan*nSamp.

void sendLVCommand(PKT_TYPES pkt_type, io_struct_t ios);
// Write command packet to Labview. 


bool LVCmdAvail(char &cmdChar, long int params[], int &N);
  // Checks serial port from Labview.
  // Expects a single alpha character, comma, digits, comma, digits, semicolon.
  // Read a string until we get ";" termination character
  // e.g. "S,8,125;"
  // Params are passed by reference so we can pass values back to the caller
  // Returns true if we got a complete command.
  // ***** NOTE: By convention, size of params array must be at least 10 *****

bool LVCmdAvail_float(char &cmdChar, float params[], int &N);
// As above but for float parameters.




//                              ADS1299_data_acquisition_V2.ino

//  Bill Goolsby, Emory Cell Biology Electronics and Machining Core
//
//  Prototype for Chris Rodgers, 3/28/23.
//  Adapted from 3/10/23 version of Plessey_Human_V3.ino.
//  Uses a Textured Walkway PC board wired for the signals needed by the ADS1299
//  evaluation board.  Board has a Teensy 4.0.

//  Works with Labview program:
//  Labview\Chris Rodgers\ADS1299 data acquisition\ADS1299 data acq.llb\Test ADS1299 acquire.vi

// ***** Compilation procedure *****
// Set board to Teensy 4.0, 600MHz.  Use Arduino 2.3.2.

// Operational notes
// - If you change from one Teensy to another you must reboot the computer.
//   Just exiting and restarting Labview doesn't work.
//   This must have something to do with the changing COM port.

// Serial1 is used for debug output.

// 3/10/23 - Made some variables in loop() static.  Now works using U and S commands.
// 3/29 - removed time sync code.
// 3/31 - Data acq is working.
//   Works up to 1KHz sample rate w/ buffer size = 256, 4KHz w/ 1024, 8KHz w/ 2048,
//   16KHZ mostly w/ 3072, 16KHz w/ no visible glitches at 4000.
//   (I'm referring to ADC_BUFFER_SIZE_SAMPLES here)
//   ADS1299 board data input setup:  I disconnected the CH1 +/- input jumpers.
//   I connected (-) input to the VREFP testpoint TP3 (2.000V).  I connected (+)
//   input to the fcn. gen + output and fcn. gen GND to AGND.
//   Fcn. gen. settings: 1Vpp, zero offset, 6Hz.  
//   The signal appears in LV centered at 7.0V, max 7.5, min 6.5.
//   SPI freq = 6MHz.
//   -----
//   Changed:  delay in ISR from 10 to none, delay in sample_loop from 15 to 2,
//     removed write to g_data in ISR since this wasn't used.
//   ISR requires 43us out of 62.5us period.
//   sample_loop requires 0.5 - 1.0ms out of 31.25ms period.
// 4/10 - Added trigger input (trig_in) as interrupt on rising edge of pin 32.  
//   When triggered, the current sample number is put into LVHdrSpecs.auxSamples
//   for LV to read out.  Several changes required in LV where auxSamples was used -
//   in all those cases the value 0 is substituted.
// 4/27 - Corrected POWER_LED wiring, connected DATA_LED and toggle DATA_LED in sample_loop().
//   Added Inquire response in checkSampleLoopCommands().
// 5/04 - In ADS_ISR, changed /CS to remain low for all 27 bytes, and changed
//   sign-extend values to 0x0800000 and 0xFF000000.  Now the pk-pk range is correct
//   but there is an offset of -7.0V.  I had put -7 offset in LV Scale Data; once I
//   removed that, offset goes to zero.
// --------------------------------- V2 -------------------------------------
// 6/13 - V2.  Changes:
//   Gain is now settable per channel.  Gain array always has 8 entries.
//   U command now receives 8 gain entries.  Param array size increased to 20.
//   Trigger lockout time = 100ms.
// --------------------------------- V2B ------------------------------------
// 7/15/24 - Added impedance testing, fcn. config_impedance_test(), called from
//   setup_adc().


// To do:
// - PKT_ERROR is not being handled by LV, e.g. in S command without previous U.
// - is it necessary to copy the buffer in sample_loop?  it doesn't take much time, so don't worry about it.
// - sometimes when you stop LV, Teensy does not get the X command
// - change trig_in lockout to 50ms



#include <SPI.h>
//#include <MemoryFree.h>

#include "math.h" // for Pi and sin()

#include "src\LVDataHeader_V3\LVDataHeader_V3.h"



const String ProgName = "ADS1299 V2B";
const String dateStr  = "240719";

// *************************** TEXTURED WALKWAY V1 BOARD ****************************

//ADS1299 Teensy control lines
const int ADS_CS     =    33;  //neg true
const int ADS_RESET  =    24;  //neg true
const int ADS_START  =    25;  //pos true
const int ADS_DRDY   =    26;  //Falling transition signals data ready

const int DB0        =    27;  //debug output for ISR
const int DB1        =    28;  //debug output for sample_loop()
const int DATA_LED   =    29;  //active high
// reserve const int SPKR_IN    =    30;  //trigger input, normally low
const int POWER_LED  =    31;  //active high
const int TRIG_IN    =    32;  //trigger input, normally low

// Connections to EXTRA IDC16 on PC board
//	Fcn		Teen.	IDC16
//	SCK				 9
//	MOSI			10
//	MISO			11
//	CS		33		12
//	RESET	24		13
//	START	25		14
//	DRDY	26		15
//	GND				 8
//	TRIG_IN	32		 3  //goes through HC14 Schmidt trigger inverter.


//ADS1299 software commands - datasheet p.40
const byte SDATAC = 0x11;	// stop data continuous
const byte START  = 0x08;	// start conversion
const byte STOP   = 0x0A;	// stop conversion
const byte RDATAC = 0x10;	// enable read data continuous
const byte RREG   = 0x20;	// read register; must be ORed with x xxxx start address
							// 2nd byte = #registers to read-1, 000x xxxx
const byte WREG   = 0x40;	// write register; same comments as above


// ADS1299 REGISTER ADDRESSES
const int CONFIG1 = 0x01; // p.44
const int CONFIG2 = 0x02;
const int CONFIG3 = 0x03;
const int CH1SET  = 0x05;
const int CH2SET  = 0x06;
const int CH3SET  = 0x07;
const int CH4SET  = 0x08;
const int CH5SET  = 0x09;
const int CH6SET  = 0x0A;
const int CH7SET  = 0x0B;
const int CH8SET  = 0x0C;
// Related to impedance testing
const int LOFF        = 0x04; // p.49
const int LOFF_SENSP  = 0x0F; // p.53
const int LOFF_SENSN  = 0x10; // p.54
const int LOFF_FLIP   = 0x11; // p.55

const int NCHANS = 8; // #Hardware channels in device; always acquire this many
const int BIT_DEPTH = 24;
// Register addresses for individual channels
const byte CHREGS[NCHANS] = {CH1SET, CH2SET, CH3SET, CH4SET, CH5SET, CH6SET, CH7SET, CH8SET};



// ************************ END BOARD DEFS *******************************


const uint16_t DEVICE_ID     = 1; 
const uint16_t BOARD_VERSION = 1; //PCB version

const bool SERIAL_DEBUG = true;   //This goes over Serial2 so it can coexist with USB serial
const bool LLDB         = true;   //low-level debug; can be used at low sample rates
const int  PARAM_ARRAY_SIZE = 20; //for incoming LV commands.  Increased for new U command.

const int TRIG_IN_LOCKOUT = 100; // ms to lock out until next allowable trigger

unsigned long startTime, stopTime;

enum state_t {s_idle, s_sampling, s_error};
state_t g_State = s_idle;  //must start in idle state
bool g_setup_cmd_OK = false;  // Setup (U) must be received before Start

const int DI_SIZE = 5; //size of debug info array
struct debug_info_t;
void shift_in_debug_data(bool init, debug_info_t di, debug_info_t dia[]);
void print_debug_info(int location, int j, int rb_index, debug_info_t di[]);
struct debug_info_t {
	int  j;
	int  rbi_after_fill;
	int  pkt_seq;
	int  pkt_index;
	bool last_pkt;
};

// Specs of the ADS1299 ADC
// tclk, the master clock, is 2.048MHz, 488ns period.
// This is separate from the SPI clock.
const int16_t  ADC_REF_VOLTAGE_MV  =  4500;
const int16_t  POS_FS              =  ADC_REF_VOLTAGE_MV; //mV at GAIN = 1.  Sent to LV in Q command
const int16_t  NEG_FS              = -ADC_REF_VOLTAGE_MV; //mV at GAIN = 1
const int16_t  ZERO_OFFSET         =     0; //to be subtracted from data in LV
const uint16_t ADC_BIT_DEPTH       = BIT_DEPTH;
const uint16_t ADC_MAX_CHANNELS    =     8;
const uint16_t ADC_MIN_SAMPLE_RATE =   250;
const uint16_t ADC_MAX_SAMPLE_RATE = 16000;

// Max SPI speed for ADS1299 is 20MHz; use less for testing.
// Max sampling rate = 16KHz, period = 62.5us
// Theoretical SPI rate needed = 16Ksps * 27 bytes * 8 bits/byte = 3.46Mbps
// *** Don't make this too much faster than needed.  6MHz works.
SPISettings ADSSPISetting(6000000, MSBFIRST, SPI_MODE1); //clock idles low, capture data on falling edge

union buffer_union_t {  //Allows buffer to be viewed as int32 or uint8 for Serial.write
	int32_t  i[ADC_BUFFER_SIZE_SAMPLES]; // signed 32-bit int
	uint8_t  b[ADC_BUFFER_SIZE_BYTES];   // 4*size above - bytes for transmission
};

/*** GLOBAL ADC VARIABLES FOR THE ISR - MUST BE DECLARED VOLATILE ***/
volatile bool    g_bufferReady = false;
// Buffers for the samples to send over USB
volatile buffer_union_t buffer1;
volatile buffer_union_t buffer2;
// Buffer pointers - we want these to point to int32_t, i.e. samples, not bytes
volatile int32_t *volatile frontBuffer = &buffer1.i[0];
volatile int32_t *volatile rearBuffer  = &buffer2.i[0];
// sample counter, used in ISR
static volatile uint32_t g_buffCounter = 0; // Keeps track of the samples in buffer (ISR)
// Why is this static and others aren't?
volatile uint8_t  g_adc_nchan = NCHANS; //ISR #channels to sample. set from g_as.nchan_sample
volatile int32_t  g_recv32; //ISR signed 32-bit SPI received value
volatile uint16_t g_adci;   //counter for use in ISR
volatile bool     g_isr_enable_sampling = false; //whether ISR stores data or just returns
volatile int32_t  g_data[ADC_MAX_CHANNELS]; //signed; for testing - results for all channels
// trig_in added 4/10/23
volatile bool     g_trig_in_received = false;
volatile int16_t  g_trig_in_sampleCount = 0;
volatile uint32_t g_trig_in_time = 0; // allow 1st trig immediately
/*** END GLOBAL ADC VARIABLES ***/


unsigned long g_pktCount = 0;   //counter and timer for keeping track transfer info
unsigned long g_loopct   = 0;  
unsigned long g_bufferct = 0;
unsigned long g_pkt_sets = 0;
unsigned long g_xmit_start;     //time when sample_loop starts


#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

int freeMemory() {
  char top;
#ifdef __arm__
  return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
  return &top - __brkval;
#else  // __arm__
  return __brkval ? &top - __brkval : &top - __malloc_heap_start;
#endif  // __arm__
}


void setup(void) {
	//io_struct_t ios;
	
	setup_pins();
	// Start serial output, 12Mbit/sec no matter what over teensy USB
	Serial.begin(115200);  //USB
	delay(1500);
	Serial1.begin(57600);  //CPU Pins 0/1 to 4-pin debug header
	delay(1500);
	char cc = 'x'; 
	while ( Serial.available() ) {  //flush USB buffer
		cc = Serial.read();
	}
	if (cc == 'x') Serial1.println(); //to quiet compiler about cc
	Serial1.print(F("Program "));  Serial.println(ProgName);

	SPI.begin(); //do this before setup_adc
	//setup_adc(ios, true, gainArray); // this will be done in U command
	Serial1.println("Setup complete");
} // end setup()


void setup_pins() {
// Set pin modes for ADS1299 interface. 
	pinMode(ADS_CS,         OUTPUT);  // ADC chip select, active low
	digitalWrite(ADS_CS,    HIGH);
	pinMode(ADS_RESET,      OUTPUT);
	digitalWrite(ADS_RESET, HIGH);
	pinMode(ADS_START,      OUTPUT);
	digitalWrite(ADS_START, LOW);
	pinMode(ADS_DRDY,       INPUT_PULLUP);
	pinMode(TRIG_IN,        INPUT);

	pinMode(DB0,            OUTPUT);
	digitalWrite(DB0,       LOW);
	pinMode(DB1,            OUTPUT);
	digitalWrite(DB1,       LOW);
	pinMode(POWER_LED,      OUTPUT);
	digitalWrite(POWER_LED, HIGH); //turn on
	pinMode(DATA_LED,       OUTPUT);
	digitalWrite(DATA_LED,  LOW); //turn off
}


void cmd_write(byte cmd) {  // Write a byte to the chip
	SPI.beginTransaction(ADSSPISetting);
		digitalWrite(ADS_CS, LOW);
		SPI.transfer(cmd);
		digitalWrite(ADS_CS, HIGH);
		delayMicroseconds(10);
	SPI.endTransaction();
}


byte cmd_write_read(byte cmd) {  // Write a byte and return result
	SPI.beginTransaction(ADSSPISetting);
		digitalWrite(ADS_CS, LOW);
		byte b = SPI.transfer(cmd);
		digitalWrite(ADS_CS, HIGH);
		delayMicroseconds(10);
	SPI.endTransaction();
	return b;
}

byte readreg(byte addr, bool long_delay) {  // Read a single register
	cmd_write_read(SDATAC);
	if (long_delay) delay(50); else delay(1);
	cmd_write_read(SDATAC);
	// Keep /CS low for all transfers
	SPI.beginTransaction(ADSSPISetting);
		digitalWrite(ADS_CS, LOW);
		SPI.transfer(RREG | addr);
		delayMicroseconds(5);
		SPI.transfer(0); // 0 means 1 register
		delayMicroseconds(5);
		byte reg_read = SPI.transfer(0);
		digitalWrite(ADS_CS, HIGH);
		delayMicroseconds(10);
	SPI.endTransaction();
	return reg_read;
	}


void writereg(byte addr, byte value) {  // Write a single register
	cmd_write_read(SDATAC);
	delay(1);
	cmd_write_read(SDATAC);
	delay(1);
	// Keep /CS low for all transfers
	SPI.beginTransaction(ADSSPISetting);
		digitalWrite(ADS_CS, LOW);
		SPI.transfer(WREG | addr);
		delayMicroseconds(5);
		SPI.transfer(0); // 0 means 1 register
		delayMicroseconds(5);
		SPI.transfer(value);
		digitalWrite(ADS_CS, HIGH);
		delayMicroseconds(10);
	SPI.endTransaction();
	}


void test_data_setup(byte check[]) {
// Set up chip to acquire data from the internal test signal at 250 sps
	byte i, r;

	cmd_write_read(SDATAC);
	delay(10);
	writereg(CONFIG1, 0x96); // 250 sps - p.46
	writereg(CONFIG2, 0xD0); // internal test
	writereg(CONFIG3, 0xE0); // internal ref.
	for (int n=0; n<NCHANS; n++) {
		i = WREG | CHREGS[n];
		r = RREG | CHREGS[n];
		writereg(i, 0x05); // set test signal as channel input
		check[n] = readreg(r, false);
	}
}

byte rate_code(int samprate_hz) { // p.46
	byte c;
	switch (samprate_hz) {
		case 16000: c = 0;  break;
		case  8000: c = 1;  break;
		case  4000: c = 2;  break;
		case  2000: c = 3;  break;
		case  1000: c = 4;  break;
		case   500: c = 5;  break;
		case   250: c = 6;  break;
		default: c = 6;
	}
	return c;
}

byte gain_code(int gain) { // p.50 in datasheet
	byte c;
	switch (gain) {
		case  1: c = 0;  break;
		case  2: c = 1;  break;
		case  4: c = 2;  break;
		case  6: c = 3;  break;
		case  8: c = 4;  break;
		case 12: c = 5;  break;
		case 24: c = 6;  break;
		default: c = 0;
	}
	c = c << 4; // shift up so lsb is in position 4
	return c;
}


void power_up_sequence() {
	// Power-up sequence
	Serial1.println("    power_up_sequence");
	SPI.beginTransaction(ADSSPISetting);
		digitalWrite(ADS_CS, LOW);
		digitalWrite(ADS_RESET, LOW);
		digitalWrite(ADS_START, LOW);
		delay(3); //ms
		digitalWrite(ADS_RESET, HIGH); // 6/15 - moved reset high to here from below
		digitalWrite(ADS_CS, HIGH);
		delay(6);
		//digitalWrite(ADS_RESET, HIGH);
		delay(5);
	SPI.endTransaction();
}

void startup_sequence() {
	Serial1.println("    startup_sequence");
	cmd_write(SDATAC); // stop data continuous
	delay(1);
	digitalWrite(ADS_RESET, LOW);
	delay(2);
	digitalWrite(ADS_RESET, HIGH);
	delay(5);
	cmd_write(SDATAC);
	delay(3);
	cmd_write(STOP);
	delay(1);
}



void config_rate_and_gain(int samprate_hz, bool test_signal, int gainArray[]) {
// Configure registers for sample rate and gain.
// Uses gainArray for gain values.
// Rate code is 3 bits, 000 to 110 - p.46.
// Gain code is 3 bits, 000 to 110 - p.50
	byte cmd, gainBits;

	Serial1.println("    Config_rate_and_gain");
	Serial1.println(String("    Rate = ") + String(samprate_hz));
	cmd_write_read(SDATAC);
	delay(10);
	cmd_write_read(STOP);
	delay(2);
	writereg(CONFIG1, 0x90 | rate_code(samprate_hz)); // p.46
	Serial1.println(String("    CONFIG1 = ") + String(readreg(CONFIG1, false),HEX));
	writereg(CONFIG2, 0xC0); // no test signal
	Serial1.println(String("    CONFIG2 = ") + String(readreg(CONFIG2, false),HEX));
	writereg(CONFIG3, 0xE0); // internal ref.
	Serial1.println(String("    CONFIG3 = ") + String(readreg(CONFIG3, false),HEX));

	for (int n=0; n<NCHANS; n++) {
		cmd = WREG | CHREGS[n];
		if (test_signal) { // p.50
			gainBits = 0x05;
		} else {
			gainBits = gain_code(gainArray[n]); // electrode input, gain in bits 6:4
		}
		writereg(cmd, gainBits);
		Serial1.println(String("    Gain[") + String(n) + String("] = ") + String(gainArray[n]) +
			String("   bits ") + String(gainBits,HEX));
		Serial1.println(String("    ") + String(cmd,HEX) + 
			String("    CHxREG[") + String(n) + String("] =  ") + 
			String(readreg(CHREGS[n], false),HEX));
	}
}

void config_impedance_test(uint16_t flags) {
	// See p.44:
	// Set channel in LOFF_SENSP and N, addr 0Fh and 10H
	// frequency:  FLEAD_OFF [1:0] in LOFF reg, addr 04h
	// reverse the current in LOFF_FLIP bits, addr 11h
	// Flag bits:
	//   15 14 13 12    11 10  9  8    7  6  5  4    3  2  1  0
	//                     fl  ch ch   ch cu cu f    f  i
	//   10  - flip direction of current
	//   9:7 - channel
	//   6:5 - current
	//   4:3 - freq
	//   2   - impedance test active
	uint8_t v;
	bool impTest = (flags & 0x04) == 0x04;
	if (impTest) {
		uint8_t  freq    = (flags & 0x18)  >> 3;
		uint8_t  current = (flags & 0x60)  >> 5;
		uint16_t chan    = (flags & 0x380) >> 7;
		bool     flip    = (flags & 0x400) >> 10;
		
		v = (current << 2) | freq; // current: bits 3:2, freq: bits 1:0
		writereg(LOFF, v);
		v = 1 << chan; //  "1" in one of bits 7:0
		writereg(LOFF_SENSP, v);
		writereg(LOFF_SENSN, v);
		if (flip) { 
			v = 1 << chan; //  "1" in one of bits 7:0
			writereg(LOFF_FLIP, v);
		}
		Serial1.println("Impedance registers:");
		Serial1.println(String("  Freq = ") + String(freq) +
			String("   Crnt = ") + String(current) +
			String("   Chan = ") + String(chan) +
			String("   Flip = ") + String(flip));
	} else {  // no test; set default values
		v = 0;
		writereg(LOFF,       v);
		writereg(LOFF_SENSP, v);
		writereg(LOFF_SENSN, v);
		writereg(LOFF_FLIP,  v);
	}
}


void setup_adc(io_struct_t &ios, bool set_defaults, int gainArray[]) {
// For ADS1299.
// Sets up ADC except for enabling the ISR - this is done in start_adc();

	if (set_defaults) {
		// set default adc specs - these will eventually come in a 
		// PKT_STREAMING_CONTROL packet from receiver.
		ios.as.nchan_acquire	= NCHANS;
		ios.as.nchan_keep		= NCHANS;
		ios.as.sample_rate_hz	= 250; //per channel
		ios.as.gain             = 1;
		ios.as.flags            = 0;
		ios.as.packet_scheme	= 0;
		ios.as.bit_depth		= BIT_DEPTH;
		ios.as.sample_averages	= 1;
		ios.as.device_id		= DEVICE_ID;
	}
	g_adc_nchan = ios.as.nchan_acquire; //this is used by ISR
	g_pktCount = 0;
	
	digitalWrite(ADS_CS, HIGH);
	delay(1);

	// Software reset then set up registers
	Serial1.println("setup_adc");
	power_up_sequence();
	startup_sequence();
	bool test_signal = byte(ios.as.flags) & 0x01;
	config_rate_and_gain(ios.as.sample_rate_hz, test_signal, gainArray);
	config_impedance_test(ios.as.flags);

	g_isr_enable_sampling = false; //ISR shouldn't store data yet

	Serial1.println("setup_adc complete");
} //end setup_adc


void start_adc(io_struct_t ios) {
// Starts ADC running, using specs in ios.as.
	Serial1.println("    start_adc");
	g_bufferReady = false;
	g_buffCounter = 0;
	g_adc_nchan   = ios.as.nchan_acquire; //used by ISR
	
	g_trig_in_received = false;
	attachInterrupt(digitalPinToInterrupt(TRIG_IN), TRIG_isr, RISING);

	attachInterrupt(digitalPinToInterrupt(ADS_DRDY), ADS_isr, FALLING);
	cmd_write(RDATAC); // enable read data continuous
	cmd_write(START);  // start conversions - must send this after RDATAC

	g_isr_enable_sampling = true;
	Serial1.println("    Start ADC - sampling enabled");
}


void stop_adc(void) {
	Serial1.println("    stop_adc");
	cmd_write(SDATAC); // stop data continuous
	cmd_write(STOP);
	detachInterrupt(digitalPinToInterrupt(ADS_DRDY));
	g_isr_enable_sampling = false;
}


/* ADC Interrupt Service Routine for ADS1299.
 * Called on the falling edge of DRDY.
 * Reads 27 bytes, the first 3 of which are status bytes.
 * The next 24 are 8 channels of 3 bytes each.
 * The appropriate buffer is then selected and samples are placed in the buffer.
 * All variables must be global and volatile!
 * 5/04 - changed /CS to remain low for all 27 bytes, and changed
 *        sign-extend values to 0x0800000 and 0xFF000000.
 */
void ADS_isr() {
	digitalWriteFast(DB0, HIGH);  // debug pin to measure ISR time
	if (g_isr_enable_sampling) {
		SPI.beginTransaction(ADSSPISetting);

		digitalWriteFast(ADS_CS, LOW);
		// Ignore 3 status bytes
		g_recv32 = int32_t(SPI.transfer(0)); // MSB
		g_recv32 = (g_recv32 << 8) | int32_t(SPI.transfer(0));
		g_recv32 = (g_recv32 << 8) | int32_t(SPI.transfer(0)); // LSB
		//digitalWriteFast(ADS_CS, HIGH); // 5/04
		//delayMicroseconds(2);  //was 10

		for (g_adci=0; g_adci < NCHANS; g_adci++) {  // read NCHANS channels
			//digitalWriteFast(ADS_CS, LOW); // 5/04
			g_recv32 = int32_t(SPI.transfer(0)); // MSB
			g_recv32 = (g_recv32 << 8) | int32_t(SPI.transfer(0));
			g_recv32 = (g_recv32 << 8) | int32_t(SPI.transfer(0)); // LSB
			//digitalWriteFast(ADS_CS, HIGH); // 5/04
			// Sign extend - set bits 24 to 31 the same as bit 23
			if (g_recv32 &  0x0800000) {
				g_recv32 |= 0xFF000000; 
			}

			rearBuffer[g_buffCounter++] = g_recv32; //store data & increment index
			// Switch buffer if necessary
			if (g_buffCounter >= ADC_BUFFER_SIZE_SAMPLES) {
				// Swap buffers.	WNG: changed syntax to elim. compile errors
				if (frontBuffer == &buffer1.i[0]) {
					frontBuffer = &buffer2.i[0];
					rearBuffer	= &buffer1.i[0];
				} else {
					frontBuffer = &buffer1.i[0];
					rearBuffer	= &buffer2.i[0];
				}
				g_buffCounter = 0; // moved these 2 lines to after buffer pointers are swapped
				g_bufferReady = true; // signal sample_loop() to transmit buffer
			}
		} //end for
		digitalWriteFast(ADS_CS, HIGH); // 5/04
		
		SPI.endTransaction();
	} //end if
	digitalWriteFast(DB0, LOW);
} //end ADS_isr


void TRIG_isr() {
// ISR called on rising edge of TRIG_IN
	//digitalWriteFast(DB2, HIGH);  // debug pin
	g_trig_in_received = true;
	g_trig_in_sampleCount = int16_t(g_buffCounter / NCHANS);
	g_trig_in_time = millis();
	//digitalWriteFast(DB2, LOW);
} //end TRIG_isr



void checkSampleLoopCommands(io_struct_t ios, bool &done) {
// Handle commands from LV that may come in while sample_loop is running.
	long int params[PARAM_ARRAY_SIZE];
	int n_params;
	char cmdChar;
	bool cmdReady;

	cmdReady = LVCmdAvail(cmdChar, params, n_params);
	if (cmdReady) {
		if (cmdChar == 'X') { //Stop
			sendLVCommand(PKT_MSG_ACK, ios);
			done = true;
			if (SERIAL_DEBUG) Serial1.println("Got Stop command, exiting sample_loop");
		} else {
			done = false;
		}
		if (cmdChar == 'I') { //Inquire
			if (SERIAL_DEBUG) {
				Serial1.println("Respond to I command");
			}
			ios.qs.device_id	 = DEVICE_ID;
			ios.qs.board_version = BOARD_VERSION;
			sendLVCommand(PKT_RECV_PRESENT, ios);
			done = false;
		}
	}
}


void sample_loop(io_struct_t ios) {
// 3/09/23 - Moved setup_adc() to setup_sampling().

	bool done = false;
	long int loopct = 0;
	int16_t trig_in_sc = -1;
	uint32_t next_led = millis();
	bool led_on = true;

	int32_t recv_buf[ADC_BUFFER_SIZE_SAMPLES]; 

	Serial1.println(String("Enter sample_loop"));
	start_adc(ios);
	Serial1.println(String("Transmitting..."));
	g_xmit_start = millis();
	
	//trig input
	g_trig_in_received = false;
	g_trig_in_sampleCount = 0;
	
	while (!done) {
		if (g_bufferReady) {  // sample buffer is full - send packet
			if (LLDB) {
				Serial1.print("Pkt ct   = ");
				Serial1.println(g_pktCount);
				Serial1.print("Free mem =  ");
				Serial1.println(freeMemory());
			}
			digitalWrite(DB1, HIGH);  //LED on for time it takes to transmit buffer
			g_bufferct++;
			
			// Copy frontbuffer to recv_buf
			for (int i=0; i < ADC_BUFFER_SIZE_SAMPLES; i++) { 
				recv_buf[i] = frontBuffer[i];
			}
			
			int NSAMP = ADC_BUFFER_SIZE_SAMPLES / ios.as.nchan_acquire;

			if (g_trig_in_received && millis() >= g_trig_in_time) {
				// set trigger input sample count
				trig_in_sc = g_trig_in_sampleCount;
				g_trig_in_received = false;
				g_trig_in_time += TRIG_IN_LOCKOUT;
			} else {
				trig_in_sc = -1;
			}
			//sendLVData(buf[],       nChan,           nSamp,    total_samples, 
			//           RECV_BUF_SAMPLES_ALL_PACKETS was replaced with ADC_BUFFER_SIZE_SAMPLES
			sendLVData32(recv_buf, ios.as.nchan_acquire, NSAMP, ADC_BUFFER_SIZE_SAMPLES, 
				// sampRate,               seqNum                 trig_in sample count);
				ios.as.sample_rate_hz, uint8_t(g_pktCount % 256), trig_in_sc);  //see LVDataHeader
				
			g_pktCount++;
			g_bufferReady = false;

			digitalWrite(DB1, LOW);
			//if ((g_pktCount % 100) == 0) {
			//	if (LLDB) Serial1.println(String("  P ") + String(loopct));
			//}
		} else {  //sample buffer not ready - do other tasks
			delayMicroseconds(2); // was 15.  max sample rate 16KHz = 62.5us
			checkSampleLoopCommands(ios, done); //check for incoming commands
			if (millis() >= next_led) {
				led_on = !led_on;
				digitalWrite(DATA_LED, led_on);
				next_led += 1000;
			}
		} //end if g_bufferReady
		loopct++;
	} //end while
} // end sample_loop




// **************************************** TESTS ***************************

void test_ADS1299() {
// Non-Labview test:  sends serial data to Arduino serial plotter.
// Can send to Serial or Serial1, since this doesn't involve Labview.
// Prints Ch0 only, but this can be changed by modifying the if (j == 0) code.

	io_struct_t ios;

	ios.as.nchan_acquire	=   NCHANS;
	ios.as.nchan_keep		=   NCHANS;
	ios.as.sample_rate_hz	=   250; //per channel
	ios.as.packet_scheme	=   0;
	ios.as.bit_depth		=   BIT_DEPTH;
	ios.as.sample_averages	=   1;
	ios.as.device_id		=   DEVICE_ID;
	ios.as.gain             =   1;
	ios.as.flags            =   1; //test signal
	
	int gainArray[NCHANS] = {1,1,1,1,1,1,1,1};

	setup_adc(ios, false, gainArray);
	start_adc(ios);
	Serial1.println("Start serial test");
	Serial1.println("");
	
	const int loops = 100;

	while (1) {
		if (g_bufferReady) {  // sample buffer is full
			g_bufferct++;
			int bufIndex = 0; //index into frontBuffer
			//int loops = ADC_BUFFER_SIZE_SAMPLES / g_adc_nchan;
			//Serial.println(String("Buffer ") + String(g_bufferct));
			for (int l=0; l<loops; l++) {
				for (int j=0; j < NCHANS; j++) {
					if (j == 0) {
						Serial.print("Ch1:");
						Serial.print(frontBuffer[bufIndex]);
						Serial.println();
						// Serial plotter requires a comma between different
						// variables.
					}
					bufIndex++;
				}
				//Serial.println();
			}
			//Serial.println();
			g_bufferReady = false;
		}
	}
} //end test_ADS1299


void test_sine_data(io_struct_t ios) {
// Generates sine wave data; doesn't use ADC.  This is used to test interface to 
// Labview program:
//   Labview\Chris Rodgers\ADS1299 data acquisition\ADS1299 data acq.llb\Test ADS1299 acquire.vi

	Serial1.println();
	Serial1.println("Start sine data test");

	const int BUFFER_SIZE_SAMPLES = 360;
	int32_t recv_buf[BUFFER_SIZE_SAMPLES]; 
	const int LOOPS = BUFFER_SIZE_SAMPLES;
	uint8_t pktCount = 0;
	bool done = false;

	while (!done) {
		for (int i=0; i<LOOPS; i++) {
			int32_t s = int32_t(1000 * sin(i * M_PI / 180.0));
			//Serial1.println(String("    s = ") + String(s));
			for (int j=0; j < NCHANS; j++) {
				if (j == 0) {
					recv_buf[i] = s;
					//Serial.print("Ch1:");
					//Serial.print(s);
					//Serial.println();
					// Serial plotter requires a comma between different
					// variables.
				}
			}
		} // end for i
		Serial1.println(String("    pktCount = ") + String(pktCount));
		//sendLVData(buf[],  nChan,  nSamp,    total_samples,    samprate, seqNum  trig_in
		sendLVData32(recv_buf, 1,      LOOPS,    BUFFER_SIZE_SAMPLES, 1000, pktCount, -1);
		pktCount++;
		checkSampleLoopCommands(ios, done); //check for incoming commands
		delay(125);
	} // end while
	Serial1.println("End sine data test");
	Serial1.println();

} //end test_sine_data




// ****************************************  LOOP ***************************


void loop(void) {
// Adapted from state_recv_v8.ino
// ***** NOTE: By convention, size of params array must be at least 10 *****

	// Static items for loop() - static needed between U and S commands
	static io_struct_t ios;
	static int nChan, sampRate, flags, gain;

	char cmdChar;
	long int serParam[PARAM_ARRAY_SIZE];
	int gainArray[NCHANS];
	int  nParams;
	bool cmdReady;

	// Put test routines here
	//test_ADS1299();
	//test_fake_data();

	switch(g_State) {
		case s_idle:
			//SetLED(Blue);
			cmdReady = LVCmdAvail(cmdChar, serParam, nParams);
			//if (SERIAL_DEBUG && cmdReady) {
			//	Serial1.println(String("cmdReady: ") + String(cmdChar));
			//}
			if (cmdReady) {
				switch (cmdChar) {
					// *** Note:  All commands must reply to Labview via sendLVCcommand()
					// *** with either packet type (success) or PKT_ERROR (failure).
					case 'U':  //Set Up sampling
						if (nParams == 11) {
							nChan       = serParam[0];
							sampRate    = serParam[1];
							flags       = serParam[2];
							//gain        = serParam[3];
							for (int i = 3; i<(3+NCHANS); i++) {
								gainArray[i-3] = serParam[i];
							}
							if (SERIAL_DEBUG) {
								Serial1.print("Got Setup cmd, nChan "); Serial1.print(nChan);
								Serial1.print("  sampRate "); Serial1.print(sampRate);
								Serial1.print("  flags ");    Serial1.println(flags);
								Serial1.print("  Gains  ");
								for (int i=0; i<NCHANS; i++) {
									Serial1.print(gainArray[i]);  Serial1.print("  ");
								}
								Serial1.println();
							}
							ios.as.nchan_acquire	= nChan;
							ios.as.sample_rate_hz	= sampRate;
							ios.as.nchan_keep		= ios.as.nchan_acquire;
							ios.as.packet_scheme	= 0;
							ios.as.bit_depth		= BIT_DEPTH;
							ios.as.sample_averages	= 1;
							ios.as.gain             = 0;
							ios.as.flags            = flags;
							sendLVCommand(PKT_MSG_ACK, ios);
							setup_adc(ios, false, gainArray);
							g_State = s_sampling;
							//SetLED(Green);
							g_setup_cmd_OK = true;
							// sample_loop(ios);
							// //now sampling.  Sampling continues until we get Stop from LV
							// //...
							// //receive_loop returned because we got a Stop command from LV
							sendLVCommand(PKT_IDLE, ios);
							g_State = s_idle;
						} else { //wrong # params
							sendLVCommand(PKT_ERROR, ios);
							Serial1.println(String("Error: U recvd. wrong # params: ") + String(nParams));
						}//endif nParams == 3
						break;

					case 'S':  // Start sampling
						if (g_setup_cmd_OK) { // Setup must have been sent before Start
							if (SERIAL_DEBUG) {
								Serial1.print("Got Start cmd, nChan "); Serial1.print(nChan);
								Serial1.print("  sampRate "); Serial1.print(sampRate);
								Serial1.print("  gain ");    Serial1.println(gain);
							}
							sendLVCommand(PKT_MSG_ACK, ios);
							g_State = s_sampling;
							//SetLED(Green);
							
							sample_loop(ios);  // sample_loop calls start_sampling()
							//test_sine_data(ios);

							//now sampling
							//...
							//sample_loop returned because we got a Stop command from LV
							stop_adc();
							sendLVCommand(PKT_IDLE, ios);
							g_State = s_idle;
							g_setup_cmd_OK = false;
							if (SERIAL_DEBUG) {
								Serial1.println("Got Stop cmd");
								Serial1.println();  Serial1.println();
							}
						} else {
							sendLVCommand(PKT_ERROR, ios);
							Serial1.println(String("Error: S recvd. but not setup cmd"));
						}
						break;

					case 'I': //Inquire receiver present
						if (SERIAL_DEBUG) {
							Serial1.print("Respond to I command: ");
							Serial1.print(DEVICE_ID);
							Serial1.print("  ");
							Serial1.println(BOARD_VERSION);
						}
						ios.qs.device_id	 = DEVICE_ID;
						ios.qs.board_version = BOARD_VERSION;
						sendLVCommand(PKT_RECV_PRESENT, ios);
						g_State = s_idle;
						break;

					case 'Q': // Query specs
						// Fill ios.qs fields to return to LV
						ios.qs.device_id       = DEVICE_ID;
						ios.qs.board_version   = BOARD_VERSION;
						ios.qs.nchan_max       = ADC_MAX_CHANNELS;
						ios.qs.sample_rate_max = ADC_MAX_SAMPLE_RATE;
						ios.qs.bit_depth       = ADC_BIT_DEPTH;
						ios.qs.pos_FS          = POS_FS; // at gain = 1
						ios.qs.neg_FS          = NEG_FS;
						ios.qs.zero_offset     = ZERO_OFFSET;
						Serial1.print("    PKT_QUERY_SPECS  ");  Serial1.print(ios.qs.device_id);
						Serial1.print("  ");  Serial1.print(ios.qs.board_version);
						Serial1.print("  ");  Serial1.print(ios.qs.nchan_max);
						Serial1.print("  ");  Serial1.print(ios.qs.sample_rate_max);
						Serial1.print("  ");  Serial1.println(ios.qs.bit_depth);
						sendLVCommand(PKT_QUERY_SPECS, ios);
						g_State = s_idle;
						break;
					case 'X': // Stop - just reply OK
						sendLVCommand(PKT_MSG_ACK, ios);
						Serial1.println("Got Stop command in main loop");
						break;
					default:
						Serial1.println(String("Default case, got : ") + String(cmdChar));
						break;
				} //end switch cmdChar
			} //end if cmdReady
			break;
		case s_sampling:
			Serial1.println(String("Sampling state"));
			break;
		case s_error:
			Serial1.println(String("Error state"));
			break;
	} //end switch g_state
	delay(1);  // was 1

} //end loop

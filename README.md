# ppg-hr-rr-app
Code for a Simple Biosensor mobile app using PPG 


#Software side Here

#Hardware side 

Components requires 
Microcontroller : NodeMcu
Display (with/without touch) : 2.4" TFT Touch display
heart rate sensor : pulse sensor
Powerchord : powerbank with usb 
Breadboard (MINI) : BreadBoard

#SERVER 
Mail:
Pass:
https://mqtt.one/
username:
pass:
Access keys:
Publish and subscribing channels/topics:

#Connections
Wiring for 2.4" TFT Display (ILI9341 SPI)
TFT Pin	NodeMCU Pin	Notes
VCC	3.3V	Do NOT use 5V
GND	GND	Common ground
CS	D8	Chip select
RESET	D4	You can also tie to 3.3V
DC / RS	D3	Data/command
MOSI	D7	SPI MOSI
MISO	D6	SPI MISO
SCK	D5	SPI clock
LED	3.3V	Backlight
T_CS (Touch CS)	D2	Touch screen select
T_IRQ	No connection	Optional
T_DO	D6	Same MISO
T_DIN	D7	Same MOSI
T_CLK	D5	Same SCK

Power: Use 3.3V only. The ILI9341 works at 3.3V.




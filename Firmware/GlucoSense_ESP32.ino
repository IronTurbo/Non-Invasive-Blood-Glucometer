#include <Wire.h>
#include "MAX30105.h"
#include <NimBLEDevice.h> // Modern, lightweight BLE library for ESP32

// BLE UUIDs for Nordic UART Service
#define SERVICE_UUID           "6e400001-b5a3-f393-e0a9-e50e24dcca9e" // UART service UUID
#define CHARACTERISTIC_UUID_RX "6e400002-b5a3-f393-e0a9-e50e24dcca9e" 
#define CHARACTERISTIC_UUID_TX "6e400003-b5a3-f393-e0a9-e50e24dcca9e" // ESP32 sends data here

NimBLEServer *pServer = NULL;
NimBLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;
bool oldDeviceConnected = false;

MAX30105 particleSensor;


// Callback for connection status
class MyServerCallbacks: public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer) {
      deviceConnected = true;
      Serial.println("Phone connected!");
    };
    void onConnect(NimBLEServer* pServer, ble_gap_conn_desc* desc) {
      deviceConnected = true;
      Serial.println("Phone connected!");
    };
    void onDisconnect(NimBLEServer* pServer) {
      deviceConnected = false;
      Serial.println("Phone disconnected.");
    }
};

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("Initializing MAX30102...");

  // Initialize Sensor
  if (particleSensor.begin() == false) {
    Serial.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  // Same settings you used for data collection
  particleSensor.setup(35, 4, 2, 100, 215, 4096);

  // Initialize BLE
  Serial.println("Initializing BLE...");
  NimBLEDevice::init("GlucoSense_ESP32"); // This name appears on the phone
  
  pServer = NimBLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  NimBLEService *pService = pServer->createService(SERVICE_UUID);

  // NimBLE uses NIMBLE_PROPERTY::NOTIFY and automatically handles descriptors
  pTxCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID_TX,
                      NIMBLE_PROPERTY::NOTIFY
                    );

  // NimBLE uses NIMBLE_PROPERTY::WRITE
  NimBLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                       CHARACTERISTIC_UUID_RX,
                       NIMBLE_PROPERTY::WRITE
                     );

  pService->start();

  // Simplified NimBLE Advertising setup
  NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
  pAdvertising->setName("GlucoSense_ESP32");
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  Serial.println("BLE Started! Waiting for phone to connect...");
}

void loop() {
  // Sync connection state directly from NimBLE hardware
  bool isConnected = pServer->getConnectedCount() > 0;

  // Read sensor
  uint32_t red = particleSensor.getRed();
  uint32_t ir = particleSensor.getIR();

  // If connected, send over BLE
  if (isConnected) {
    // Format: "Red,IR\n"
    String payload = String(red) + "," + String(ir) + "\n";

    // Set value and notify the phone
    pTxCharacteristic->setValue(payload.c_str());
    pTxCharacteristic->notify();
    
    // Print to Serial monitor so you can see it's actually sending!
    Serial.print("Sent to phone: ");
    Serial.print(payload);
  }

  // Handle disconnecting
  if (!isConnected && oldDeviceConnected) {
      delay(500); 
      pServer->startAdvertising(); // restart advertising
      Serial.println("Restarted BLE advertising");
      oldDeviceConnected = isConnected;
  }
  
  // Handle connecting
  if (isConnected && !oldDeviceConnected) {
      Serial.println("Phone connected! (Detected by loop)");
      oldDeviceConnected = isConnected;
  }

  delay(10);
}
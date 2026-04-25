#include <Wire.h>
#include <WiFi.h>  // مكتبة الوايفاي الجديدة

// ==========================================
// ⚠️ إعدادات الوايفاي (لازم تغير التلات سطور دول) ⚠️
// ==========================================
const char* ssid = "Staff";         // 1. اكتب اسم شبكة الوايفاي بتاعتك هنا
const char* password = "Sta@2025"; // 2. اكتب باسوورد الوايفاي هنا
const char* serverIP = "10.107.19.25";        // 3. اكتب الـ IP بتاع اللاب توب اللي جبناه من الـ cmd
const uint16_t serverPort = 8080;            // البورت اللي البايثون مستنيه (ماتغيروش)

WiFiClient client; // كائن الاتصال باللاب توب

// --- تعريفات الحساس زي ما هي ---
const int MPU_addr = 0x68;
double pitchInput, rollInput, yawInput, altitudeInput;
double xAcc, yAcc, zAcc, xGyro, yGyro, zGyro;
double currentGyroMillis, previousGyroMillis, deltaGyroTime;
double pitchInputAcc, rollInputAcc, yawInputAcc;
double pitchInputGyro, rollInputGyro, yawInputGyro;

double rollGyroOffset = 0, pitchGyroOffset = 0, yawGyroOffset = 0;
double rollAccOffset = 0, pitchAccOffset = 0, yawAccOffset = 0;

void setup() {
  Wire.begin(8, 9);  // SDA = 8, SCL = 9
  Serial.begin(115200);

  // --- الاتصال بالوايفاي ---
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("=====================================");

  // --- معايرة الحساس (زي ما هي) ---
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1B);
  Wire.write(0x10);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1C);
  Wire.write(0x10);
  Wire.endTransmission(true);
  
  currentGyroMillis = millis();

  Serial.println("Calibrating Accelerometer... Please wait. Keep Sensor Still!");
  if (rollAccOffset == 0) {
    for (int i = 0; i < 200; i++) {
      Wire.beginTransmission(MPU_addr);
      Wire.write(0x3B);
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_addr, 6, true);

      xAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;
      yAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;
      zAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;

      pitchAccOffset += (atan((yAcc) / sqrt(pow((xAcc), 2) + pow((zAcc), 2))) * RAD_TO_DEG);
      rollAccOffset += (atan(-1 * (xAcc) / sqrt(pow((yAcc), 2) + pow((zAcc), 2))) * RAD_TO_DEG);

      if (i == 199) {
        rollAccOffset = rollAccOffset / 200;
        pitchAccOffset = pitchAccOffset / 200;
      }
      delay(3); 
    }
  }

  Serial.println("Calibrating Gyroscope... Please wait. Keep Sensor Still!");
  if (rollGyroOffset == 0) {
    for (int i = 0; i < 200; i++) {
      Wire.beginTransmission(MPU_addr);
      Wire.write(0x43);
      Wire.endTransmission(false);
      Wire.requestFrom(MPU_addr, 6, true);

      xGyro = (int16_t)(Wire.read() << 8 | Wire.read());
      yGyro = (int16_t)(Wire.read() << 8 | Wire.read());
      zGyro = (int16_t)(Wire.read() << 8 | Wire.read());

      rollGyroOffset += yGyro / 32.8 ;
      pitchGyroOffset += xGyro / 32.8;
      yawGyroOffset += zGyro / 32.8;
      
      if (i == 199) {
        rollGyroOffset = rollGyroOffset / 200;
        pitchGyroOffset = pitchGyroOffset / 200;
        yawGyroOffset = yawGyroOffset / 200;
      }
      delay(3);
    }
  }
  Serial.println("Calibration Complete! Starting readings...\n");
  delay(1000); 
}

void loop() {
  // --- التأكد من الاتصال باللاب توب ---
  if (!client.connected()) {
    Serial.println("Connecting to Laptop Server...");
    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Connection failed! Retrying in 1 second...");
      delay(1000);
      return; // ارجع أول اللوب لحد ما يتصل
    }
    Serial.println("Connected to Laptop successfully!");
  }

  // --- حسابات الحساس (زي ما هي) ---
  previousGyroMillis = currentGyroMillis;
  currentGyroMillis = millis();
  deltaGyroTime = (currentGyroMillis - previousGyroMillis) / 1000.0;

  Wire.beginTransmission(MPU_addr);
  Wire.write(0x43);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr, 6, true);
  
  xGyro = (int16_t)(Wire.read() << 8 | Wire.read());
  yGyro = (int16_t)(Wire.read() << 8 | Wire.read());
  zGyro = (int16_t)(Wire.read() << 8 | Wire.read());

  xGyro = (xGyro / 32.8) - pitchGyroOffset;
  yGyro = (yGyro / 32.8) - rollGyroOffset;
  zGyro = (zGyro / 32.8) - yawGyroOffset;

  pitchInputGyro = xGyro * deltaGyroTime ;
  rollInputGyro = yGyro * deltaGyroTime;
  yawInputGyro = zGyro * deltaGyroTime;

  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr, 6, true);
  
  xAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;
  yAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;
  zAcc = (int16_t)(Wire.read() << 8 | Wire.read()) / 4096.0 ;

  pitchInputAcc = (atan((yAcc) / sqrt(pow((xAcc), 2) + pow((zAcc), 2))) * RAD_TO_DEG) - pitchAccOffset;
  rollInputAcc = (atan(-1 * (xAcc) / sqrt(pow((yAcc), 2) + pow((zAcc), 2))) * RAD_TO_DEG) - rollAccOffset;

  rollInput = 0.98 * (rollInput + rollInputGyro) + 0.02 * (rollInputAcc);
  pitchInput = 0.98 * (pitchInput + pitchInputGyro) + 0.02 * (pitchInputAcc);
  yawInputAcc = atan2((sin(rollInput) * cos(pitchInput) * xAcc + sin(pitchInput) * yAcc + cos(rollInput) * cos(pitchInput) * zAcc), sqrt(pow(sin(rollInput) * sin(pitchInput) * xAcc - cos(rollInput) * sin(pitchInput) * zAcc, 2) + pow(cos(pitchInput) * xAcc, 2))) - 1;
  yawInput = 0.98 * (yawInput + yawInputGyro) + 0.02 * (yawInputAcc);
  
  // --- إرسال الداتا للاب توب عن طريق الوايفاي (أرقام بس بدقة عالية) ---
  client.print(xAcc, 5);
  client.print(",");
  client.print(yAcc, 5);
  client.print(",");
  client.print(zAcc, 5);
  client.print(",");
  client.print(rollInput, 5);
  client.print(",");
  client.print(pitchInput, 5);
  client.print(",");
  client.println(yawInput, 5); // println بتضيف سطر جديد في الآخر عشان البايثون يعرف إن القراءة خلصت
  
  // 20ms delay to keep the 50Hz reading rate
  delay(10);
}

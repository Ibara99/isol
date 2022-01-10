#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h> 
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <SoftwareSerial.h>

//deklarasi variabel
SoftwareSerial DataSerial (12,13); //D6=12; D7=13
#define LED_BUILTIN 2 

//millis sebagai ganti delay; jaga-jaga mcu ter-reset sendiri
unsigned long prevMillis = 0;
const long interval =  10000; //10 detiik
int c = 6; //dikirim pada ke 6
//ampun om, ini logicnya ampas emang

String arrData[2];

//konfig WiFi
const char* ssid = "Anemoria";
const char* password = "Adminnya1?";

//Web/Server address to read/write from 
const char *host = "pengembangan-tugas-akhir-s1.ibaraasro.repl.co";
//const char *host = "iot-template.mulaabmulyo.repl.co";
const int httpsPort = 443;  //HTTPS= 443 and HTTP = 80

//SHA1 finger print of certificate use web browser to view and copy
//const char fingerprint[] PROGMEM = "F2 A1 1C 61 5D DA 06 28 71 26 EB E6 78 66 F1 B5 4B E7 54 DE";
// VALID HINGGA 31 JANUARI 2022
const char fingerprint[] PROGMEM = "68 BF 67 D6 D0 9C 16 0B C8 5A D0 B4 07 6B 72 CE F8 F1 B4 9A";

WiFiClient espClient;

void setup() {
  Serial.begin(9600);
  DataSerial.begin(9600);
  WiFi.mode(WIFI_OFF);        //Prevents reconnection issue (taking too long to connect)
  delay(1000);
  WiFi.mode(WIFI_STA);        //Only Station No AP, This line hides the viewing of ESP as wifi hotspot
  
  //konek ke wifi
  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    digitalWrite(LED_BUILTIN, LOW); //selama ga konek, led mati
  }
  //apabila konek, led baru idup
  digitalWrite(LED_BUILTIN, HIGH);

}

void loop() {
  //konfig millis
  unsigned long currMillis = millis(); //waktu sekarang
  if(currMillis - prevMillis >= interval){
    //update prevmillis
    prevMillis = currMillis;

    //baca dari arduino (dari dataserial)
    String data = "";
    while(DataSerial.available()>0){
      data += char(DataSerial.read());
    }
    //buang spasi
    data.trim();

    //cek data
    if(data != ""){
      //format yg dikirim dari arduino ph#salinitas
      //parse data dulu
      char delimiter = '#';
      int index = 0; //untuk memasukkan data ke array
      for (int i=0; i<data.length(); i++){
        if (data[i] != delimiter)
          arrData[index] += data[i];
        else
          index++;
      }
      //cek pembacaan
      if(index == 1){
        Serial.println(arrData[0]);
        Serial.println(arrData[1]);
        Serial.println();
        if (c%(6) == 0){
          //String msg2 = String(arrData[0])+"-"+String(arrData[1]);
          //client.publish(topic, (char*) msg2.c_str());
          String msg2 = "ph="+String(arrData[0])+"&sal="+String(arrData[1]);
          httpPOST(msg2);
          c = 1;
        }else{
          c++;
        }
      }
      //reset arrData
      arrData[0] = "";
      arrData[1] = "";    
    }
    //req data ke arduino
    //Ya ini keyword yg disamakan di program arduino
    DataSerial.println("Ya");  
  }
//  Serial.println("tes");
}

void httpPOST(String msg){
  BearSSL::WiFiClientSecure httpsClient;    //Declare object of class WiFiClient
  httpsClient.setInssecure();
  
  
  Serial.println(host);

//  Serial.printf("Using fingerprint '%s'\n", fingerprint);
//  httpsClient.setFingerprint(fingerprint);
//  httpsClient.setTimeout(15000); // 15 Seconds
  delay(1000);
  
  Serial.print("HTTPS Connecting");
  int r=0; //retry counter
  while((!httpsClient.connect(host, httpsPort)) && (r < 30)){
      delay(100);
      Serial.print(".");
      r++;
  }
  if(r==30) {
    Serial.println("Connection failed");
  }
  else {
    Serial.println("Connected to web");
  }
  
  String getData, Link;
  
  //POST Data
  Link = "/api/addData";

  Serial.print("requesting URL: ");
  Serial.println(host);
  /*
   POST /post HTTP/1.1
   Host: postman-echo.com
   Content-Type: application/x-www-form-urlencoded
   Content-Length: 13
  
   say=Hi&to=Mom
    
   */
  //String msg = "h=7&t=17";
  httpsClient.print(String("POST ") + Link + " HTTP/1.1\r\n" +
               "Host: " + host + "\r\n" +
               "Content-Type: application/x-www-form-urlencoded"+ "\r\n" +
               "Content-Length: "+ msg.length() + "\r\n\r\n" +
               msg + "\r\n" +
               "Connection: close\r\n\r\n");

  Serial.println("request sent");
  Serial.println(msg);
}

/*  
 * D4 AF 4F C8 A3 4C AB 1D 01 4A 74 79 DE 68 A1 A6 76 DB E0 DB
 */
 

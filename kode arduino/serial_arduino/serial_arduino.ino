//konfig sensor
#define PH_PIN A5
#define SALINITAS_PIN A0

//deklarasi variabel
float ph, salinitas;

void setup() {
  Serial.begin(9600);
}

void loop() {
  //kirimData();
  //baca permintaan dari mcu
  String minta = "";
  while (Serial.available() > 0) {
    minta += char(Serial.read());
  }
  //buang spasi
  minta.trim();
  //cek
  if (minta == "Ya") {
    kirimData();
  }
  //reset variable
  minta = "";
  delay(1000);
}

void kirimData() {
  //baca sensor
  ph = analogRead(PH_PIN);
  ph = toPH(ph);
  delay(2000);
  salinitas = analogRead(SALINITAS_PIN);

  // untuk ngirim data ke nodemcu
  // template: "phsekian#salinitassekian"; nanti tinggal di parse
  String data = String(ph) + "#" + String(salinitas);
  //kirim data ke mcu
  Serial.println(data);
}

float toPH(float x) {
  //nanti ini taruh arduino aja, biar sekalian baca data;
  //mcu cuma buat kirim data doang
  //y=-0.1524*X + 160.0
  //0.005179*X + 4.053 //aslinya
  return 0.005179 * x + 4.053;
}

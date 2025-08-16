/*
  Este código é responsável por ler uma matriz 8x8 enviada via Serial, fazer o parsing (interpretação) 
  dessa string para extrair os valores float e armazená-los corretamente em uma matriz do tipo float no Arduino.

  Após a leitura e construção da matriz, os dados são processados e enviados para um modelo de rede neural 
  embarcado (via TensorFlow Lite for Microcontrollers), que realiza a **classificação de padrões** com base 
  nesses valores. O modelo pode ser usado, por exemplo, para reconhecer gestos, detectar anomalias ou 
  identificar comportamentos com base em sensores.

  A matriz deve ser enviada como uma matriz em String JSON com 8 listas contendo 8 valores, por exemplo:

    "[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                         ...
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]"

  Etapas:
  ------------------------------------------------------------
  1. Leitura dos dados da porta Serial até encontrar '\n'
  2. Remoção dos colchetes iniciais e finais ('[' e ']')
  3. Divisão da string em partes separadas por vírgula
  4. Conversão de cada pedaço em float
  5. Armazenamento dos valores na matriz float[8][8]
  6. Execução do modelo de IA para realizar a classificação
  ------------------------------------------------------------

  Esse processo é necessário pois os dados enviados via Serial chegam como texto,
  e precisam ser convertidos em valores numéricos para uso no Arduino.

  *** AUTOR: JOÃO VICTOR PELEGRINO ***

*/

#include <ArduTFLite.h>
#include "CNN_model.h"

const int tamMatriz = 8;
constexpr int tensorArenaSize = 50 * 1024;
alignas(16) byte tensorArena[tensorArenaSize];

const char* LABELS[] = { "Normal", "Def_Rolo", "Def_Pista_Ext" };
#define NUM_LABELS (sizeof(LABELS) / sizeof(LABELS[0]))

float matrizEntrada[tamMatriz][tamMatriz];
String inputString = "";
bool stringComplete = false;

// Variáveis para medir tempo
unsigned long startParse, endParse;
unsigned long startInfer, endInfer;
unsigned long startTotal, endTotal;

void setup() 
{
  Serial.begin(9600);
  while (!Serial);

  pinMode(LED_BUILTIN, OUTPUT);

  if (!modelInit(model, tensorArena, tensorArenaSize))
  {  
    Serial.println("Falha na inicializacao do modelo!");
    while (true);
  }
}

void loop() 
{
  if (stringComplete)
  {                
    Serial.println("String recebida:");
    Serial.println(inputString);
    
    if (parseMatrizFromString(inputString))
    {  
      Serial.println("\nMatriz convertida com sucesso.");
      runInferenceAndBlinkLed();
    } 
    else 
    {
      Serial.println("Erro ao converter matriz. Verifique o formato da matriz enviada.");
    }

    inputString = "";
    stringComplete = false;

    Serial.println("\nAguardando próxima matriz via serial...");
    Serial.println("FIM");
  }
}

void serialEvent() 
{
  while (Serial.available())            
  {            
    char inChar = (char)Serial.read();  
    if (inChar == '\r') continue;       
    if (inChar == '\n')                 
    {               
      stringComplete = true;
      break;
    }
    inputString += inChar;              
  }
}

bool parseMatrizFromString(String str) 
{
  startParse = micros();  // Início do parsing

  str.replace("[", "");
  str.replace("]", "");
  str.replace("{", "");
  str.replace("}", "");
  str.replace("\t", "");
  str.replace("\n", "");
  str.replace("\r", "");
  str.replace(" ", "");

  int expectedValues = tamMatriz * tamMatriz;
  float valores[expectedValues];

  int start = 0;
  int idx = 0;

  while (idx < expectedValues) 
  {
    int commaIndex = str.indexOf(',', start);
    String valorStr;

    if (commaIndex == -1) 
      valorStr = str.substring(start);
    else 
      valorStr = str.substring(start, commaIndex);

    if (valorStr.length() == 0) return false;

    valores[idx] = valorStr.toFloat();
    idx++;

    if (commaIndex == -1) break;
    start = commaIndex + 1;
  }

  if (idx != expectedValues) return false;

  for (int i = 0; i < tamMatriz; i++) 
    for (int j = 0; j < tamMatriz; j++) 
      matrizEntrada[i][j] = valores[i * tamMatriz + j];

  endParse = micros(); // Fim do parsing
  Serial.print("Tempo de parsing (us): ");
  Serial.println(endParse - startParse);

  return true;
}

void runInferenceAndBlinkLed() 
{
  startTotal = micros(); // Início total

  Serial.println("\nMatriz para inferência:");
  for (int i = 0; i < tamMatriz; i++) 
  {
    for (int j = 0; j < tamMatriz; j++) 
    {
      Serial.print(matrizEntrada[i][j], 6);
      Serial.print("\t");
    }
    Serial.println();
  }
  Serial.println();

  for (int r = 0; r < tamMatriz; r++) 
    for (int c = 0; c < tamMatriz; c++) 
      modelSetInput(matrizEntrada[r][c], r * tamMatriz + c);

  Serial.println("Executando inferência...");

  startInfer = micros();  // Início da inferência
  if (!modelRunInference())  
  {         
    Serial.println("Falha na inferência!");
    return;
  }
  endInfer = micros();    // Fim da inferência

  Serial.print("Tempo de inferência (us): ");
  Serial.println(endInfer - startInfer);

  Serial.println("\nResultados:");
  for (int i = 0; i < NUM_LABELS; i++) 
  {
    float prob = modelGetOutput(i);
    Serial.print(LABELS[i]);
    Serial.print(": ");
    Serial.print(prob * 100, 2);
    Serial.println("%");
  }
  
  float maiorValor = 0.0;
  int indiceMaior = -1;

  for (int i = 0; i < NUM_LABELS; i++) 
  {
    float prob = modelGetOutput(i);
    if (prob > maiorValor)
    {
      maiorValor = prob;
      indiceMaior = i;
    }
  }

  if (maiorValor >= 0.85) 
  {
    Serial.print("Classe detectada com confiança >= 85%: ");
    Serial.println(LABELS[indiceMaior]);

    switch (indiceMaior) 
    {
      case 0: piscarLED(1); break;
      case 1: piscarLED(2); break;
      case 2: piscarLED(3); break;
    }
  } 
  else 
  {
    Serial.println("Nenhuma classe acima do limiar de 85%.");
  }

  endTotal = micros(); // Fim total
  Serial.print("Tempo total (us): ");
  Serial.println(endTotal - startTotal);
}

void piscarLED(int vezes) 
{
  for (int i = 0; i < vezes; i++) 
  {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
  }
}

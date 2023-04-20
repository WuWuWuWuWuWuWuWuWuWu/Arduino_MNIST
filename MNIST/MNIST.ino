// #include "aug_1_uint8.tflite.h"
// #include "aug_2_uint8.tflite.h"
// #include "aug_3_uint8.tflite.h"
#include "aug_4_uint8.tflite.h"

#include "image_data.h"
#include "image_list.h"

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  int inference = 120;

	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* input = nullptr;
	TfLiteTensor* output = nullptr;
  
	constexpr int kTensorArenaSize = 1024 * 50;
	uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

		// Map the model into a usable data structure. This doesn't involve any
		// copying or parsing, it's a very lightweight operation.
		model = tflite::GetModel(tflite_model);
		if(model->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);

    // This pulls in all the operation implementations we need.
		// NOLINTNEXTLINE(runtime-global-variables)
		static tflite::AllOpsResolver resolver;

		// Build an interpreter to run the model with.
		static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
		interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
		TfLiteStatus allocate_status = interpreter->AllocateTensors();
		if(allocate_status != kTfLiteOk) MicroPrintf("AllocateTensors() failed");

		// Obtain pointers to the model's input and output tensors.
		input = interpreter->input(0);
		output = interpreter->output(0);
}

void loop() {
  uint8_t correct = 0;
  double time = 0;
  double accuracy_list[120] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double latency_list[120] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  
  Serial.println("index\t\tlabel\tpredict\taccuracy\tlatency");
	for(int i= 0; i < inference; i++) {
    double start_time = millis();
		// Place the quantized input in the model's input tensor
		for(int j= 0; j < 784; j++) input->data.uint8[j] = image_data[i][j];

		// Run inference, and report any error
		TfLiteStatus invoke_status = interpreter->Invoke();
		if(invoke_status != kTfLiteOk) MicroPrintf("Invoke failed\n");
    uint8_t predict = 0;
		uint8_t max = 0;
		for(int j= 0; j < 10; j++) {
      if(output->data.uint8[j] > max) {
        predict = j;
        max = output->data.uint8[j]; 
      }
    }        
    time += (millis() - start_time);
    if(predict == image_list[i]) correct += 1;
    Serial.println("\t" +String(i + 1) + "\t\t" + String(image_list[i]) + "\t\t" + String(predict) + "\t\t" + String(double(correct) / double(i + 1)) + "\t\t" + String(time / double(i + 1)));
    accuracy_list[i] = double(correct) / double(i + 1);
    latency_list[i] = time / double(i + 1);
		delay(100);
	}

  Serial.println("accuracy:\t" + String(double(correct ) / double(inference)));
  Serial.print("\t[");
  for(int i= 0; i  < 120; i++) {
    Serial.print(String(accuracy_list[i]));
    if(i != 119) Serial.print(", ");
  }
  Serial.println("]");

  Serial.println("latency:\t" + String(time / double(inference)));
  Serial.print("\t[");
  for(int i= 0; i  < 120; i++) {
    Serial.print(String(latency_list[i]));
    if(i != 119) Serial.print(", ");
  }
  Serial.println("]");
  delay(100000);
}

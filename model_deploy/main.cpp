#include "mbed.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "uLCD_4DGL.h"
#include "mbed_rpc.h"
#include "stm32l475e_iot01_accelero.h"
#include "mbed_events.h"
#include "math.h"

#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
// The gesture index of the prediction
int gesture_index;
int angle;
int THangle = 0;
int mode; //0 is gestureUI, 1 is tilt angle detection, 2 is RPC.
int16_t pDataXYZ_[3] = {0};
DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
MQTT::Client<MQTTNetwork, Countdown> *ptr_client;

BufferedSerial pc(USBTX, USBRX);
Thread t1;
Thread t2;
InterruptIn user_bt(USER_BUTTON);

uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

void uLCD_print();
int PredictGesture(float* output);
void Gesture_UI();
RPCFunction rpc1(&Gesture_UI, "Gesture_UI");
void angle_detection();
RPCFunction rpc2(&angle_detection, "angle_detection");
void find_gesture();
void find_angle();
void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client);

void uLCD_print(){
  if(mode != 0) return;

  if(gesture_index == 0){
    THangle = 30;
  }else if(gesture_index == 1){
    THangle = 60;
  }else if(gesture_index == 2){
    THangle = 90;
  }else;
  //print "gesture_index" on uLCD.
  uLCD.background_color(0xFFFFFF);
  uLCD.cls();
  uLCD.text_width(1); //2X size text
  uLCD.text_height(1);
  uLCD.textbackground_color(WHITE);
  uLCD.color(BLUE);
  uLCD.printf("\nthreshold angle:\n"); //Default Green on black text    
  uLCD.text_width(4); //4X size text
  uLCD.text_height(4);
  uLCD.color(GREEN);
  uLCD.locate(1,2);
  uLCD.printf("%2d", THangle);
  return;
}

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void Gesture_UI(){
  mode = 0;
  myled1 = 1;
  myled2 = 0;
  //start a thread.
  printf("start gesture UI.\n");
  t1.start(find_gesture);
}

void find_gesture(){
  //keep on finding gesture_index
  while(true){
    //predict gesture.
    if(mode == 0){
      // Whether we should clear the buffer next time we fetch data
      bool should_clear_buffer = false;
      bool got_data = false;

      // Set up logging.
      static tflite::MicroErrorReporter micro_error_reporter;
      tflite::ErrorReporter* error_reporter = &micro_error_reporter;

      // Map the model into a usable data structure. This doesn't involve any
      // copying or parsing, it's a very lightweight operation.
      const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
      if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
      }

      // Pull in only the operation implementations we need.
      // This relies on a complete list of all the ops needed by this graph.
      // An easier approach is to just use the AllOpsResolver, but this will
      // incur some penalty in code space for op implementations that are not
      // needed by this graph.
      static tflite::MicroOpResolver<6> micro_op_resolver;
      micro_op_resolver.AddBuiltin(
          tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
          tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
      micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                   tflite::ops::micro::Register_MAX_POOL_2D());
      micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                   tflite::ops::micro::Register_CONV_2D());
      micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                   tflite::ops::micro::Register_FULLY_CONNECTED());
      micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                   tflite::ops::micro::Register_SOFTMAX());
      micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                   tflite::ops::micro::Register_RESHAPE(), 1);

      // Build an interpreter to run the model with
      static tflite::MicroInterpreter static_interpreter(
          model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
      tflite::MicroInterpreter* interpreter = &static_interpreter;

      // Allocate memory from the tensor_arena for the model's tensors
      interpreter->AllocateTensors();

      // Obtain pointer to the model's input tensor
      TfLiteTensor* model_input = interpreter->input(0);
      if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
          (model_input->dims->data[1] != config.seq_length) ||
          (model_input->dims->data[2] != kChannelNumber) ||
          (model_input->type != kTfLiteFloat32)) {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
      }

      int input_length = model_input->bytes / sizeof(float);

      TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
      if (setup_status != kTfLiteOk) {
        error_reporter->Report("Set up failed\n");
        return -1;
      }

      error_reporter->Report("Set up successful...\n");

      while (true) {
      
        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                     input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
          should_clear_buffer = false;
          continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed on index: %d\n", begin_index);
          continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        if (gesture_index < label_num) {
          error_reporter->Report(config.output_message[gesture_index]);
          uLCD_print(); //call uLCD print.
        }
      }
    }
    else ;
  }
  
}

void angle_detection(){
  mode = 1;
  myled2 = 1;
  myled1 = 0;
  BSP_ACCELERO_Init();
  printf("start angle_detection.\n");
  t2.start(find_angle);
}

void find_angle(){
  bool flag = true;
  bool stop = false;
  int num = 0;
  while(flag){
    if(mode == 1 && !stop){
      float xy_long, ang_pi;
      BSP_ACCELERO_AccGetXYZ(pDataXYZ_);
      xy_long = sqrt(pDataXYZ_[0]*pDataXYZ_[0] + pDataXYZ_[1]*pDataXYZ_[1]);
      ang_pi = (atan(xy_long/pDataXYZ_[2]))/3.1415926;
      angle = ang_pi*180;
      angle = (angle < 0)? angle+180 : angle;
      printf("angle: %3d \n",angle);
      if(angle > THangle) {
        printf("%3d is over threshold angle. \n",angle);
        stop = true;
        publish_message(ptr_client);
      }
  
      ThisThread::sleep_for(200ms);
    }
    if(mode == 2 ) stop = false;
  }
  /*char buff[100];
  sprintf(buff, "%d, %d, %d\n", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);*/
}

//MQTT
// GLOBAL VARIABLES
WiFiInterface *wifi;
//InterruptIn btn2(USER_BUTTON);
//InterruptIn btn3(SW3);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char* topic = "Mbed";

Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;

void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    ++arrivedcount;
}

void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client) {
    if(mode  == 2) return;
    printf("start publish message.\n");
    MQTT::Message message;
    char buff[100];
    if(mode == 0){
      sprintf(buff, "Chosed threshold angle: %d", THangle);
    }
    if(mode == 1){
      sprintf(buff, "Angle had gone over threshold: %d", angle);
    }
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);
    mode = 2; //back to RPC mode.

    //printf("rc:  %d\r\n", rc);
    //printf("Puslish message: %s\r\n", buff);
}

void close_mqtt() {
    closed = true;
}

//

int main() {
  //MQTT
  wifi = WiFiInterface::get_default_instance();
  if (!wifi) {
          printf("ERROR: No WiFiInterface found.\r\n");
          return -1;
  }
  printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
  int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
  if (ret != 0) {
          printf("\nConnection error: %d\r\n", ret);
          return -1;
  }
  NetworkInterface* net = wifi;
  MQTTNetwork mqttNetwork(net);
  MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);
  ptr_client = &client;
  //TODO: revise host to your IP
  const char* host = "172.20.10.2";
  printf("Connecting to TCP network...\r\n");
  SocketAddress sockAddr;
  sockAddr.set_ip_address(host);
  sockAddr.set_port(1883);
  printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting
  int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
  if (rc != 0) {
          printf("Connection error.");
          return -1;
  }
  printf("Successfully connected!\r\n");
  MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
  data.MQTTVersion = 3;
  data.clientID.cstring = "Mbed";
  if ((rc = client.connect(data)) != 0){
          printf("Fail to connect MQTT\r\n");
  }
  if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
          printf("Fail to subscribe\r\n");
  }

  mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
  user_bt.rise(mqtt_queue.event(&publish_message, &client));
  //btn3.rise(&close_mqtt);
    

    /*int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }

    while (1) {
            if (closed) break;
            client.yield(500);
            ThisThread::sleep_for(500ms);
    }

    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");*/
  //
  mode = 2; //RPCmode.

  uLCD.background_color(0xFFFFFF);
  uLCD.cls();
  uLCD.text_width(1); //2X size text
  uLCD.text_height(1);
  uLCD.textbackground_color(WHITE);
  uLCD.color(BLUE);
  uLCD.printf("\nthreshold angle:\n");

  //RPC mode
  char buf[256], outbuf[256];

  FILE *devin = fdopen(&pc, "r");
  FILE *devout = fdopen(&pc, "w");
  while(1) {
      memset(buf, 0, 256);
      for (int i = 0; ; i++) {
          char recv = fgetc(devin);
          if (recv == '\n') {
              printf("\r\n");
              break;
          }
          buf[i] = fputc(recv, devout);
      }
      /*if(buf[0] == '/' && buf[1] == 'G'){
        mode = 0;
      }else if(buf[0] == '/' && buf[1] == 'a'){
        mode = 1;
      }else;*/
      //Call the static call method on the RPC class
      RPC::call(buf, outbuf);      
      //printf("%s\r\n", outbuf);
  }
}

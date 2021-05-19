#ifndef CONFIG_H_
#define CONFIG_H_

// The number of labels (without negative)
#define label_num 3

struct Config {

  // This must be the same as seq_length in the src/model_train/config.py
  const int seq_length = 64;

  // The number of expected consecutive inferences for each gesture type.
  const int consecutiveInferenceThresholds[label_num] = {20, 15, 20};

  const char* output_message[label_num] = {"Selected 30 deg.\n\r","Selected 60 deg.\n\r","Selected 90 deg.\n\r"};
};

Config config;
#endif // CONFIG_H_

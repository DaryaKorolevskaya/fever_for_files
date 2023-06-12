import datetime
import logging
from artifacts.bert_repo import modeling
import os
from artifacts.bert_repo import run_classifier
import tensorflow as tf
from artifacts.bert_repo import tokenization


logging.getLogger().setLevel(logging.ERROR)

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/' + BERT_MODEL
logging.info('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))

OUTPUT_DIR = 'tmp/'
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 1.3
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
NUM_TPU_CORES = 8
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')


class FeverProcessor(run_classifier.DataProcessor):
    """Processor for the FEVER data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "eval.tsv")),  # dev
            "dev_matched")

    def get_test_examples(self, test_lines):
        """See base class."""
        return self._create_examples(test_lines, "test")

    def get_labels(self):
        """See base class."""
        return ["NOT ENOUGH INFO", "SUPPORTS", "REFUTES"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 or line == []:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = "NOT ENOUGH INFO"
            else:
                label = tokenization.convert_to_unicode(line[2])
            examples.append(
                run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BertClassifier():
    def __init__(self, model_path):
        self.init_bert_model(model_path)

    def init_bert_model(self, model_path):
        task_type = 'test'
        self.processor = FeverProcessor()
        self.label_list = self.processor.get_labels()

        self.tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

        tpu_cluster_resolver = None
        
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=OUTPUT_DIR)
        
        if task_type == 'train':
            train_examples = self.processor.get_train_examples(OUTPUT_DIR)
        else:
            train_examples = ['pad']
        num_train_steps = int(
            len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

        if task_type == 'train':
            INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
        else:
            for file in os.listdir(model_path):
                if file.endswith(".meta"):
                    INIT_CHECKPOINT = os.path.join(model_path, file[:-5])
                    break

        model_fn = run_classifier.model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
            num_labels=len(self.label_list),
            init_checkpoint=INIT_CHECKPOINT,
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)

        if task_type == 'test':
            PREDICT_BATCH_SIZE = 5
            self.estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=False,
                model_fn=model_fn,
                config=run_config,
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                predict_batch_size=PREDICT_BATCH_SIZE
            )
        else:
            self.estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=False,
                model_fn=model_fn,
                # config=run_config,
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE)

    def predict(self, test_lines):
        file_type = 'demo'
        predict_examples = self.processor.get_test_examples(test_lines)

        num_actual_predict_examples = len(predict_examples)
        PREDICT_BATCH_SIZE = 5

        predict_file = "predict.tf_record"
        run_classifier.file_based_convert_examples_to_features(predict_examples, self.label_list,
                                                               MAX_SEQ_LENGTH, self.tokenizer,
                                                               predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", PREDICT_BATCH_SIZE)

        predict_drop_remainder = True
        predict_input_fn = run_classifier.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)

        # output_predict_file = os.path.join(directory, "results_{}.csv".format(file_type))
        output_lines = []

        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
                break
            output_line = ",".join(
                str(class_probability) for class_probability in probabilities
            ) + "\n"
            output_lines.append(output_line)
            num_written_lines += 1
            
        os.remove(predict_file)
        
        return output_lines

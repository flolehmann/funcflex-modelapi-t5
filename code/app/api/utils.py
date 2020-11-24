# TF1 version
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

#tf.disable_eager_execution()


def test(path: str):
    #tf.disable_eager_execution()
    # text_generator = hub.Module(path)
    # input_sentences = ['Long Sentence 1', 'Long Sentence 2']
    # output_texts = text_generator(input_sentences)
    # tf.print(output_texts)
    # print(output_texts)
    text_generator = hub.Module(
        'https://tfhub.dev/google/bertseq2seq/roberta24_wikisplit/1')
    input_sentences = ['Long Sentence 1', 'Long Sentence 2']
    output_texts = text_generator(input_sentences)
    print(output_texts)



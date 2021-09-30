import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

# %%
# Import data then decode for py2 compat

path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

text_length = len(text)
print(f'Length of text: {text_length} words')
# %%

# Find num unique chars in datafile
vocab = sorted(set(text))
vocab_size = len(vocab)

print(f'Length of vocab: {vocab_size} characters')

# %%

# Process the text

example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab),
                                            mask_token=None)
ids = ids_from_chars(chars)
print(ids)

chars_from_ids = preprocessing.StringLookup(vocabulary = ids_from_chars.get_vocabulary(),
                                            invert=True, mask_token=None)

chars = chars_from_ids(ids)
print(chars)

tf.strings.reduce_join(chars, axis=-1).numpy()


# %%
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# %%

# Create training examples and targets
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
print(all_ids)

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

# %%

seq_length = 100
examples_per_epoch = len(text) // (seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
    print(chars_from_ids(seq))

for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())

# %%
def split_input_target(sequences):
    input_text = sequences[:-1]
    target_text = sequences[1:]
    return input_text, target_text
# %%
# Split data

split_input_target(list("Tensorflow"))

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print(f"Input:{text_from_ids(input_example).numpy()}")
    print(f"Target:{text_from_ids(target_example).numpy()}")

# %%
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000

dataset = (dataset
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)

# %%
"""
I stopped here but this is based off a TF tutorial that I found on
https://www.tensorflow.org/text/tutorials/text_generation
"""







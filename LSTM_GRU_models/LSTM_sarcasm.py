"""
Same objective of detecting sarcasm from imbd reviews but now adding LSTM/GRU
layer(s)
"""

# %%
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf

# %%
# Import data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# %%
tokenizer = info.features['text'].encoder

# Hyper-parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 10

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

# %%
# Build simple model with single LSTM

model_single_LSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_single_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_single_LSTM.summary()

# %%
# Train single LSTM model
history1 = model_single_LSTM.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
# Def plot function for acc/loss
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# %%
# PLot simple LSTM results
plot_graphs(history1, 'accuracy')
plot_graphs(history1, 'loss')

# %%
# Try simple single layer GRU model

model_single_GRU = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_single_GRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_single_GRU.summary()

# %%
# Train simple GRU model
history2 = model_single_GRU.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
# PLot simple GRU results
plot_graphs(history2, 'accuracy')
plot_graphs(history2, 'loss')

# Notice that our LSTM model performs better

# %%
# Add bi-directionality to both simple models to see effect

model_single_biLSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_single_biLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model_single_biGRU = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_single_biGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# %%
history3 = model_single_biLSTM.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
history4 = model_single_biGRU.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
# Plot single bidir-LSTM
plot_graphs(history3, 'accuracy')
plot_graphs(history3, 'loss')

# Plot single bidir-GRU
plot_graphs(history4, 'accuracy')
plot_graphs(history4, 'loss')

# %%
# Attempting multiple bidirectional LSTM layers and multi bi-GRU layers

model_multi_biLSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_multi_biLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model_multi_biGRU = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_multi_biGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# %%
history5 = model_single_biLSTM.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
history6 = model_single_biGRU.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
# Plot multi bidir-LSTM
plot_graphs(history5, 'accuracy')
plot_graphs(history5, 'loss')

# Plot multi bidir-GRU
plot_graphs(history6, 'accuracy')
plot_graphs(history6, 'loss')

# %%
# Final architecture attempt: Convo1D

model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_conv.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_conv.summary()

# %%
# Train conv1d model
history7 = model_conv.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# %%
# Plot conv1d model
plot_graphs(history7, 'accuracy')
plot_graphs(history7, 'loss')

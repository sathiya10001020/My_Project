import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example dataset (replace this with your actual dataset)
input_texts = ["Hello", "How are you?", "What's your name?"]
target_texts = ["Hi", "I'm fine, thank you!", "I'm a chatbot."]

# Tokenize input and target texts
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_seq_length = max(len(seq) for seq in input_sequences + target_sequences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

# Define the model
latent_dim = 256
num_words_output = len(target_tokenizer.word_index) + 1

encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(input_dim=len(input_tokenizer.word_index) + 1, output_dim=latent_dim, input_length=max_seq_length)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)

decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(input_dim=num_words_output, output_dim=latent_dim, input_length=max_seq_length)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
decoder_output_data = pad_sequences([sequence + [0] for sequence in target_sequences], maxlen=max_seq_length, padding='post')

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data.reshape((*decoder_output_data.shape, 1)),
    batch_size=2,
    epochs=50,
    validation_split=0.2
)

# Save the trained model
model.save('seq2seq_chatbot_model.h5')

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model_and_vocabulary(model_path, vocab_path):
    model = tf.keras.models.load_model(model_path)
    with open(vocab_path, "rb") as file:
        vocabulary = pickle.load(file)
    return model, vocabulary


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.strip().split())
    return text

def generate_sentence(model, vocabulary, input_sentence, min_words=10, temperature=1):
    reverse_vocabulary = {v: k for k, v in vocabulary.items()}
    input_sentence = preprocess_text(input_sentence)

    word_count = len(input_sentence.split())

    while word_count <= min_words:
        tokenized_sentence = [vocabulary.get(token, vocabulary['<OOV>']) for token in input_sentence.split()]
        padded_sentence = pad_sequences([tokenized_sentence], maxlen=n_max, padding='pre', value=vocabulary['<PAD>'])

        predictions = model.predict(padded_sentence, verbose=3)[0]
        predictions[0] = 0
        predictions[1] = 0
        predictions = np.array([np.power(p, 1 / temperature) for p in predictions])
        predictions = predictions / np.sum(predictions)


        next_word_idx = np.random.choice(range(len(predictions)), p=predictions)
        next_word = reverse_vocabulary.get(next_word_idx, '<OOV>')

        input_sentence += ' ' + next_word

        word_count = len(input_sentence.split())
        if word_count > min_words and input_sentence[-1] not in [".", "!", "?"]:
            min_words += 1

    return input_sentence

model_path = "LSTM_LM"
vocab_path = "vocab.pickle"
n_max = 10

model, vocabulary = load_model_and_vocabulary(model_path, vocab_path)

input_sentence = "Le futur de la France"
completed_sentence = generate_sentence(model, vocabulary, input_sentence)
print("Phrase complétée:", completed_sentence)

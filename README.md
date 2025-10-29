# Conversational AI Chatbot (Seq2Seq Model)

## 🚀 Project Overview

This project is a conversational chatbot built using an advanced deep learning model known as **Sequence-to-Sequence (Seq2Seq)**. The model was trained on a large dataset of dialogues from movies, enabling it to respond to user questions in a way that mimics natural conversation.

The model is built using TensorFlow and Keras and utilizes an **Encoder-Decoder** architecture with GRU layers.

## 📊 Dataset

* **Source:** The dataset was downloaded from Kaggle, specifically the "Cleaned Data for the Chatbot (collected from movies)" dataset.
* **Content:** The `dialogs_expanded.csv` file was used, which contains thousands of question/answer pairs extracted from movie dialogues.
* **Initial Processing:** Unnecessary columns were dropped to focus solely on the `question` and `answer` columns.

## ⚙️ Data Preprocessing

Text preprocessing is the most critical step in this project to prepare the data before feeding it to the model:

1.  **Text Cleaning:**
    * A `clean_text` function was created to clean all questions and answers.
    * Cleaning operations included: converting text to lowercase, removing brackets `[...]`, removing links (`http` and `www`), removing HTML tags, removing punctuation, and removing any words containing numbers.
2.  **Adding Start/End Tokens:**
    * A `<start>` token was added to the beginning of every sentence and an `<end>` token was added to the end.
    * These tokens are essential for the Seq2Seq model to know when to start and stop generating an answer.
3.  **Tokenization:**
    * `tf.keras.preprocessing.text.Tokenizer` was used to create a vocabulary for both questions and answers.
    * Each unique word was assigned a unique ID.
    * Every text sentence was converted into a sequence of these numerical IDs.
4.  **Padding:**
    * Since sentences have different lengths, `tf.keras.preprocessing.sequence.pad_sequences` was used to add zero-padding to the end of shorter sentences, making all sequences the same length.
5.  **Train-Test Split:**
    * The dataset was split into 90% for training and 10% for testing.
6.  **Data Pipeline:**
    * `tf.data.Dataset` was used to build an efficient data pipeline, which shuffles the data and splits it into batches of 32.

## 🤖 Model Architecture

The model is based on the **Encoder-Decoder** architecture, which consists of two main components working together:

### 1. The Encoder
The Encoder is a `tf.keras.Model` whose job is to "understand" the input question.
* **Embedding Layer:** Converts the numerical sequence (the question) into dense vectors.
* **GRU Layer:** A type of Recurrent Neural Network (RNN) that processes the sequence step-by-step. It returns the full sequence output and the final hidden state.
* **"Context Vector":** The final hidden state (the `state`) is a numerical summary, or "understanding," of the entire question. This vector is then passed to the Decoder.

### 2. The Decoder
The Decoder is a `tf.keras.Model` whose job is to "generate" the answer based on the Encoder's understanding.
* **Embedding Layer:** Receives the word generated from the previous timestep.
* **GRU Layer:** Receives the "context vector" from the Encoder as its initial hidden state.
* **Dense Layer:** A final `Dense` layer with a `softmax` activation, which predicts the *next* word in the answer from the entire vocabulary.

## 📈 Training Process

* **Optimizer:** `tf.keras.optimizers.Adam()`.
* **Loss Function:** `SparseCategoricalCrossentropy`. A mask was used to ensure the model does not calculate loss on the zero-padding tokens.
* **Teacher Forcing:**
    * To make training faster and more stable, a technique called "Teacher Forcing" was used.
    * During training, instead of feeding the model its *own* previous prediction as input for the next step, we feed it the *correct* word from the actual target answer.
* **Model Checkpointing:** The best-performing Encoder and Decoder models were saved based on the lowest validation (test) loss.

## 🔬 Inference - How the Chatbot Works

When you ask the chatbot a question, the `chatbot` function does the following:

1.  The question is cleaned and tokenized using the same process as the training data.
2.  The full question is passed through the **Encoder** to get the final "context vector."
3.  The **Decoder** begins its process:
    * It is given the `<start>` token as its first input and the "context vector" as its initial hidden state.
    * It predicts the first word of the answer.
    * **(Autoregressive Decoding):** The word it just predicted is then used as the input for the very next time step to generate the next word.
4.  This loop continues, generating one word at a time, until the model predicts the `<end>` token or the maximum sentence length (32) is reached.
5.  All the predicted words are joined together to form the final answer.

## 🛠️ Technologies Used

* Python 3
* TensorFlow & Keras
* Pandas
* NumPy
* Scikit-learn (for train/test split)
* Kaggle API

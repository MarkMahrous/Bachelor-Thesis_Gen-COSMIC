import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import tkinter as tk
from tkinter import filedialog

# Load the CSV data into a DataFrame
columns =["LOC","Cosmic"]
data = pd.read_csv('data.csv', names=columns)

data["Cosmic"]=(data["Cosmic"]=="CFP").astype(int)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

sentences = []
labels = []
sum=0
# Collect sentences and labels into the lists
for item in data.LOC:
    sentences.append(item)
    sum=sum+1
for item in data.Cosmic:
    labels.append(item)

#creating training and testing data
training_size = int(sum*0.9)
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 1000
embedding_dim = 16
max_length = 20
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# Tokenize the dataset
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

# map words to numbers
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Convert words to vectors
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert labels lists into numpy arrays
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# Compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train the model
history = model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels))


# Predict the test data
predictions = model.predict(testing_padded)
predictions = (predictions > 0.5).astype(int)

# Calculate the accuracy
#accuracy = accuracy_score(testing_labels, predictions)
#precision = precision_score(testing_labels, predictions)
#recall = recall_score(testing_labels, predictions)
#print(f'Accuracy: {accuracy}')
#print(f'Precision: {precision}')
#print(f'Recall: {recall}')
#print(classification_report(testing_labels, predictions))


# Clean each line of code
def clean_line(line):
    # remove any comments
    line = re.sub(r'//.*|#.*', '', line)
    # remove any extra spaces
    line = line.strip()
    return line

# Function to browse and select a file
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'r') as file:
            text = file.read()
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, text)

# Function to measure the CFPs in the text
def measure():
    text = text_box.get(1.0, tk.END)
    lines = text.split('\n')
    totalCFPs = 0
    for line in lines:
        line = clean_line(line)
        if line:
            sequences = tokenizer.texts_to_sequences([line])
            padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            prediction = model.predict(padded)
            threshold = 0.5
            if prediction > threshold:
                print(line)
                totalCFPs += 1
    totalCFPs_label.config(text='Total CFPs: ' + str(totalCFPs))

# Initialize Tkinter
root = tk.Tk()
root.title("Functional Size Measurement")
root.configure(bg="#f0f0f0")  # Set background color

# Create text box to display file content
text_box = tk.Text(root, height=20, width=70, bg="#ffffff", fg="#000000", font=("Arial", 12))
text_box.pack(pady=10)

# Button to browse for a file
browse_button = tk.Button(root, text="Browse", command=browse_file, bg="#008CBA", fg="#ffffff", font=("Arial", 12))
browse_button.pack(pady=5)

# Button to measure CFPs
measure_button = tk.Button(root, text="Measure", command=measure, bg="#4CAF50", fg="#ffffff", font=("Arial", 12))
measure_button.pack(pady=5)

# Label to display the result
totalCFPs_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 12, "bold"), fg="#FF0000")
totalCFPs_label.pack(pady=5)

# Run the Tkinter event loop
root.mainloop()
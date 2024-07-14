import re
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
data = pd.read_csv('data.csv', names=["LOC", "Cosmic"])

# Convert the Cosmic column to 0 and 1
data["Cosmic"] = (data["Cosmic"] == "CFP").astype(int)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Collect sentences and labels into the lists
sentences = []
labels = []
sum=0
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

# Create a pipeline
decision_tree_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])

# Train the model
decision_tree_pipeline.fit(training_sentences, training_labels)

# Make predictions
predictions = decision_tree_pipeline.predict(testing_sentences)

# Calculate the accuracy
accuracy = accuracy_score(testing_labels, predictions)
precision = precision_score(testing_labels, predictions)
recall = recall_score(testing_labels, predictions)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(classification_report(testing_labels, predictions))

# Function to measure the CFPs
def measure():
    text = text_box.get(1.0, tk.END)
    lines = text.split('\n')
    totalCFPs = 0
    for line in lines:
        line = clean_line(line)
        if line:
            prediction = decision_tree_pipeline.predict([line])
            if prediction[0] == 1:
                totalCFPs += 1
    totalCFPs_label.config(text="Total CFPs: " + str(totalCFPs))

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

# A simple AI chatbot model using TensorFlow and Keras.
# This code is heavily commented to explain the foundational principles.

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# --- Step 1: Prepare the Data ---
# This is the "brain" of our chatbot. We define a list of intents,
# which are categories of user messages. Each intent has a tag,
# a list of patterns (questions/phrases the user might say),
# and a list of responses the chatbot can give.
intents = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "Hey", "How are you", "Is anyone there?", "Hello", "Good day"],
         "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
         "context": [""]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "I am leaving", "Have a good day"],
         "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"],
         "context": [""]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"],
         "context": [""]
        },
        {"tag": "name",
         "patterns": ["What is your name?", "What's your name?", "Who are you?"],
         "responses": ["I'm a simple chatbot.", "You can call me Chabot.", "I don't have a name."],
         "context": [""]
        },
        {"tag": "about",
         "patterns": ["What do you do?", "What can you do?"],
         "responses": ["I can answer simple questions based on my training data.", "I can respond to basic queries."],
         "context": [""]
        }
    ]
}

# Now, we process this data to make it usable for our model.
# We'll create a vocabulary of all words and a list of all tags.
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = pattern.split()
        words.extend(w)
        # Add the document (a tuple of the pattern words and the tag)
        documents.append((w, intent['tag']))
        # Add the tag to our list of classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Clean up the words by removing duplicates and ignored characters
words = [w.lower() for w in words if w.lower() not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Number of documents: {len(documents)}")
print(f"Number of words: {len(words)}")
print(f"Number of classes (tags): {len(classes)}")

# Create training data
# We'll represent each sentence as a "bag of words" which is a list
# of 0s and 1s, where 1 indicates the presence of a word from our vocabulary.
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Create an output row. The output is a one-hot encoded list
    # that is 1 for the corresponding tag and 0 for all others.
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data for better training performance
np.random.shuffle(training)
training = np.array(training, dtype=object)

# Separate features (patterns) and labels (tags)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# --- Step 2: Build the Neural Network Model ---
# This is a simple feed-forward neural network with three layers.
# The network's job is to learn the relationship between the
# bag-of-words input and the correct one-hot encoded output.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
# SGD (Stochastic Gradient Descent) is an optimizer that helps the model
# learn by adjusting its weights during training.
# 'categorical_crossentropy' is a loss function used for classification tasks.
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# --- Step 3: Train the Model ---
# We train the model for 200 epochs (iterations). Each epoch, the model
# sees the entire training data and adjusts its weights to improve accuracy.
print("\nTraining the model...")
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
print("\nModel training complete!")

# --- Step 4: Use the Model to Predict a Response ---
# We need helper functions to process new user input and get a response.

# Function to convert a user sentence into a bag of words
def bag_of_words(sentence, words):
    # Tokenize the sentence and get the lowercase words
    sentence_words = sentence.lower().split()
    # Initialize the bag with zeros
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                
    return np.array(bag)

# Function to predict the intent of a sentence
def predict_class(sentence, model):
    # Get the bag of words for the sentence
    p = bag_of_words(sentence, words)
    # The model predicts the probability of each class
    res = model.predict(np.array([p]))[0]
    # We set a threshold to filter out low-confidence predictions
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a random response from a predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break
    else:
        # Fallback response if no intent is matched
        result = "I'm sorry, I don't understand that. Can you rephrase?"
    return result

# Let's test our chatbot!
print("\n--- Start the chatbot (type 'quit' to exit) ---")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    try:
        # Get the predicted intent
        ints = predict_class(user_input, model)
        # Get the response
        response = get_response(ints, intents)
        print("Bot:", response)
    except IndexError:
        print("Bot: I'm sorry, I don't understand that. Can you rephrase?")
    except Exception as e:
        print(f"An error occurred: {e}")


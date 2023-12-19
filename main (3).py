import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Data Loading and Exploration
train_df = pd.read_csv('train.csv')
print(train_df.head())

# 2. Data Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
vocab_size = len(tokenizer.word_index) + 1

max_length = 100
X = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=max_length)

# 3. Model Architecture
embedding_dim = 50

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# 4. Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Model Training
labels = train_df['target'].values
model.fit(X, labels, epochs=5, validation_split=0.2)

# 6. Model Evaluation
# Assuming you have a test dataset
test_df = pd.read_csv('test.csv')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['text']), maxlen=max_length)

# Make predictions
predictions = model.predict(X_test)

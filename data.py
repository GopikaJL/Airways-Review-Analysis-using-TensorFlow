air_df = pd.read_csv("/content/British Airways Customer Feedback Dataset.csv")
air_df.head(1)
data = air_df[["Recommended","Overall_Rating","Review","Airline Name","Review Date","Value For Money","Review_Title","Review Date"]]
data.head()
data.isna().sum()
ata.dropna(inplace = True)
data.isna().sum()
fig = plt.figure(figsize=(8,8))
plt.hist(data["Recommended"])
plt.title("Bar chart of different classes of Rexommendation",fontsize=16)
plt.legend()
plt.show()
data["length"] = data["Review_Title"].apply(len)
data.describe()
data['Review'] = data['Review'].str.lower()
data['Review_Title'] = data['Review_Title'].str.lower()
not_recommend =data[data['Recommended'] == "no"]["Review"]
not_recommend.head()
type(not_recommend)
yes_recommend =data[data['Recommended'] == 'yes']["Review"]
yes_recommend.head()
!pip install WordCloud
from wordcloud import WordCloud
negative_words = WordCloud(width=800, height=400, background_color='white').generate(' '.join(not_recommend)
)
plt.figure(figsize=(10, 5))
plt.imshow(negative_words, interpolation='bilinear')
plt.show()
positive_words = WordCloud(width=800, height=400, background_color='white').generate(' '.join(yes_recommend)
)
plt.figure(figsize=(10, 5))
plt.imshow(positive_words, interpolation='bilinear')
plt.show()
nltk.download('punkt')
nltk.download('stopwords')
def correct_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text
data['Review'] = data['Review'].apply(correct_text)
data.head()
nltk.download('punkt')
stemmer = PorterStemmer()
def stem_text(text):
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)
data['Review'] = data['Review'].apply(stem_text)
data.head()
train, test = train_test_split(data, test_size=0.33, random_state=44)
t = Tokenizer()
tf.fit_on_texts(train.Review)
vocab_size = len(t.word_index) + 1
max_length = 64
sequences_train = t.texts_to_sequences(train.Review)
sequences_test = t.texts_to_sequences(test.Review)
X_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
X_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')
y_train = train.Recommended.values
y_test = test.Recommended.values

glove_file = open('/content/glove.6B.100d.txt')
word_embeddings = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = embeddings
def converter(sentence):
    words = sentence.split()
    vector = np.zeros_like(word_embeddings[next(iter(word_embeddings))]) 
    word_count = 0
    for word in words:
        if word in word_embeddings:
            vector += word_embeddings[word]
            word_count += 1
    if word_count != 0:
        vector /= word_count 
    return vector
text_embeddings = []
for sentence in text_data:
    embedding = converter(sentence)
    text_embeddings.append(embedding)
  embedding_dim = 100
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[text_embeddings], trainable=False)
num_epochs = 10
batch_size = 1000
model = Sequential([
        embedding_layer,
        tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
tf.keras.utils.plot_model(model, show_shapes=True)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = batch_size, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)

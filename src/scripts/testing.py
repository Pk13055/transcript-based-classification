import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
model = load_model('my_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)
df = pd.read_csv("final.csv")
df1=df[(df['Abusive']==1) & (df['Hate']==1)]
X_test=df1['Text']
y_test=df1['label']
encoded_docs_test = t.texts_to_sequences(X_test)
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=25, padding='post')
ans = model.predict([padded_docs_test])
co=0
tot=0
for i in range(len(ans)):
    tot+=1
    if np.argmax(ans[i]) == 0 or np.argmax(ans[i]) == 1:
        co+=1
print(co)
print(tot)
acc = co / tot
print(acc)

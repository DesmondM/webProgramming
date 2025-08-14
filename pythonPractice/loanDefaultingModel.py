import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('loan_data.csv')

x= df[['salary', 'age', 'loan_amount', 'loan_term_months']]
y =df['paid_off']

# Features with big numerical ranges (e.g., salary = 80,000 vs. age = 40) can dominate learning, that is the job of the scaler to normalize data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history= model.fit(x_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f" Testing for accuracy ... {accuracy:.2f}")

new_person = [[50000, 30, 15000, 36]]
new_person_scaled = scaler.transform(new_person)
prediction = model.predict(new_person_scaled)
print( 'Probability of paying off the loan: ', prediction[0][0])
print( 'Predicted class: ', int(prediction[0][0])>0.5)
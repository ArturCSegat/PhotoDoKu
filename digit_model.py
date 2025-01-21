import keras
import matplotlib.pyplot as plt
import numpy as np

(trainx, trainy), (testx, testy) = keras.datasets.mnist.load_data()
trainx = keras.utils.normalize(trainx, axis=1)
testx = keras.utils.normalize(testx, axis=1)

trainy2 = np.array([np.zeros((10)) for _ in trainy])
for i, t in enumerate(trainy):
    trainy2[i][t] = 1
testy2 = np.array([np.zeros((10)) for _ in testy])
for i, t in enumerate(testy):
    testy2[i][t] = 1

model = keras.models.Sequential([
    keras.Input((28,28,1)),
    keras.layers.Conv2D(32, 5, padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(32, 5, padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(1568, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
    ])

model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.Accuracy()])

# print(model.predict(np.ones((1, 28,28))))

model.fit(trainx,  trainy2, epochs=10)
model.save("digitrecognizer.keras")

loss, accuracy = model.evaluate(testx, testy2)

print("Loss : ", loss)
print("Accuracy : ", accuracy)
model = keras.models.load_model("digitrecognizer.keras")
j = 0
count = 0
for i, t in enumerate(testy):
    if t == 7:
        count += 1
        if count > 3:
            j = i
            break

t = testx[j]
plt.imshow(t)
plt.show()
p = model.predict(np.array([t]))
print(p)
print(p.argmax())
print(testy[j])

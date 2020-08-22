from dataset_processing import DataSetProcessing
from ResNet50Model import ResNet50
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

X_train_shuffled,Y_train_shuffled,X_test_shuffled,Y_test_shuffled= DataSetProcessing.loadData()
model=ResNet50.ResNet50()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

csv_logger=CSVLogger(
    'log.csv',
    append=True,
    separator=';'
)

model.fit(X_train_shuffled, Y_train_shuffled, epochs = 20, batch_size = 64,callbacks=[csv_logger])

preds = model.evaluate(X_test_shuffled, Y_test_shuffled)
print ("Loss="+str(preds[0]))
print ("Test Accuracy="+str(preds[1]))


filename='log.csv'
df=pd.read_csv(filename,delimiter=';')
plot = df.plot.line('accuracy')
plt.show()
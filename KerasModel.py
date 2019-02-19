from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda
from keras.optimizers import Adam
from Utils import load_configs
from keras.callbacks import TensorBoard


class ValueNetworkKeras(object):
    def __init__(self):
        self.configs = load_configs("Configs_ValueNN.json")
        self.shared_weights = self.configs["shared_weights"]
        self.fully_connected = self.configs["fully_connected"]
        self.learning_rate = self.configs["learning_rate"]

        self.width = self.configs["width"]
        self.height = self.configs["height"] + 2

        self.model = self.build_model()
        self.tensorboard = TensorBoard(log_dir='TensorBoardFiles/ValueNetworkKeras', histogram_freq=0,
                          write_graph=True, write_images=False)

    def build_model(self):
        inputArray = []
        """
        for t in range(self.width):
            inputArray.append(Input(shape=(self.height,)))
            """
        inputArray = Input(shape=(24,))

        layer = inputArray

        shared_dense = Dense(32, activation='relu')
        layerArray = []
        for t in range(self.width):
            out = Lambda(lambda x: x[:,t*self.height:(t+1)*self.height])(layer)
            layerArray.append(shared_dense(out))

        layer = layerArray

        for neurons in self.shared_weights[1:]:
            shared_dense = Dense(neurons, activation='relu')
            layerArray = []
            for t in range(self.width):
                layerArray.append(shared_dense(layer[t]))
            layer = layerArray

        merged_vector = Concatenate(axis=-1)(layer)

        layer = merged_vector
        for neurons in self.fully_connected:
            layer = Dense(neurons, activation='relu')(layer)
        output_layer = Dense(1, activation='linear')(layer)
        model = Model(input=inputArray, output=output_layer)
        print(model.summary())

        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
        return model

    def train(self, X, y):
        self.model.fit(X, y,
                       batch_size=128,
                       epochs=10,
                       shuffle=True,
                       callbacks=[self.tensorboard])

#a = ValueNetworkKeras()
#a.build_model()

"""
def learn(data, labels, stacks, tiers, run_id,output_path, shared_layer_multipliers, layer_multipliers,
          batch_size, learning_rate):
    print("Training with {0} examples. Problem size: {1} tiers {2} stacks.".format(len(data), tiers, stacks))

    shared_layer_multipliers = [x for x in shared_layer_multipliers if x != 0]

    inputArray = []
    for t in range(stacks):
        inputArray.append(Input(shape=(tiers,)))

    layer = inputArray
    for i in range(len(shared_layer_multipliers)):
        shared_dense = Dense(tiers*shared_layer_multipliers[i],activation='relu')
        layerArray = []
        for t in range(stacks):
            layerArray.append(shared_dense(layer[t]))
        layer = layerArray


    merged_vector = merge(layer, mode='concat', concat_axis=-1)

    layer = merged_vector
    for i in range(len(layer_multipliers)):
        layer = Dense(tiers*stacks*layer_multipliers[i],activation='relu')(layer)
    output_layer = Dense(stacks*(stacks-1), activation='softmax')(layer)
    model = Model(input=inputArray, output=output_layer)

    logging.info("Policy Network Summary:")
    orig_stdout = sys.stdout
    log = logging.getLogger()
    sys.stdout = LoggerWriter(log.info)
    print(model.summary())
    sys.stdout = orig_stdout

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Start training policy model")
    now = datetime.datetime.now()
    model.fit(np.hsplit(data,stacks), labels,nb_epoch= 1000,batch_size=batch_size,validation_split=0.2,verbose=2,
              callbacks=[printbatch(), EarlyStopping(monitor='val_loss', patience=50, verbose=0), ModelCheckpoint(os.path.join(output_path, "models",
                            "pm_dnn_model_" + str(stacks) + "x" + str(tiers-2) +"_"+ str(now.day) + "." + str(now.month) + "." + str(now.year) + "_"
                            + str(run_id) + "_{epoch:02d}-{val_loss:.2f}"
                            + ".h5"), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])
    model.save(os.path.join(output_path, "models","pm_dnn_model_" + str(stacks) + "x" + str(tiers-2) +"_"+str(now.day)+"."+str(now.month)+"."+str(now.year)+"_"+ str(run_id)
                            +".h5"))
    logging.info("Finished training. Saved model.")
    return model"""
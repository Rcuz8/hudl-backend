# Firebase Init

import google.cloud.storage as cloud
from firebase_admin import initialize_app, credentials, storage
from os import listdir
import json
from src.main.util import io

try:
    cred = credentials.Certificate('hudl_server/fbpk.json')
    initialize_app(cred, {
        'storageBucket': 'hudpred.appspot.com'
    })
    LOCAL = False
except:
    try:
        cred = credentials.Certificate('fbpk.json')
        initialize_app(cred, {
            'storageBucket': 'hudpred.appspot.com'
        })
        LOCAL = True
    except:
        # From test directory -> svr
        cred = credentials.Certificate('../../../../hudl_server/fbpk.json')
        initialize_app(cred, {
            'storageBucket': 'hudpred.appspot.com'
        })
        LOCAL = False

# App

DEF_BUCKET = 'hudpred.appspot.com'

class bucket:

    def __init__(self, name=None):
        self.__cloud_bucket = storage.bucket(name) if name else storage.bucket()

    def upload_file(self, file, to_path):
        """Uploads a file to the bucket

        Parameters
        ----------
        file : str
           Local file ("local/path/to/file")
        to_path : str
            Bucket file path ("path/in/bucket")

        """

        blob = self.__cloud_bucket.blob(to_path)

        blob.upload_from_filename(file)

        print(
            "File {} uploaded to {}.".format(
                file, to_path
            )
        )
        return self

    def upload_dir(self, dir, to_path):
        files = listdir(dir) # Caution: breaks when using directory with nested directory.
        for file in files:
            full_path = dir + '/' + file
            self.upload_file(full_path, to_path + file)
        print('Directory Upload Complete.')
        return self

    def upload_model(self,name: str, path: str, model_id: str):
        self.upload_dir(path, 'Models/'+model_id + '/' + name + '/')

    def adjust_paths(self, dir_path: str, model_id: str, model_name: str):
        model_json_path = dir_path + '/model.json'
        try:
            jsonFile = open(model_json_path, "r")  # Open the JSON file for reading
            data = json.load(jsonFile)  # Read the JSON into the buffer
            jsonFile.close()  # Close the JSON file

            ## Working with buffered content
            weights_manifest = data["weightsManifest"]
            for wts in weights_manifest:
                paths = wts['paths']
                for i in range(len(paths)):
                    paths[i] = 'Models%2F' + model_id + '%2F' + model_name + '%2F' + paths[i]

            data["weightsManifest"] = weights_manifest
            ## Save our changes to JSON file
            jsonFile = open(model_json_path, "w+")
            jsonFile.write(json.dumps(data))
            jsonFile.close()
        except:
            io.err('Could not find I/O file.')


    def print(self):
        for blob in self.__cloud_bucket.list_blobs():
            print(blob)
        return self

# KEY = 'sampleid2'
# MDL = 'mymodel2'
# bucket().adjust_paths(dir_path=MDL, model_id=KEY, model_name=MDL)
# import keras
# import keras.layers as layers
# import tensorflowjs as tfjs
# tfjs.converters.load_keras_model()
# model = keras.models.Sequential(
#     [
#         layers.Input(shape=(10,)),
#         layers.Dense(20, activation="relu"),
#         layers.Dense(20, activation="relu"),
#         layers.Dense(10, activation="relu")
#     ]
# )
# # inputs = keras.Input(shape=(10,))
# # dense = layers.Dense(20, activation="relu")
# # x = dense(inputs)
# # x = layers.Dense(20, activation="relu")(x)
# # outputs = layers.Dense(10)(x)
# # model = keras.Model(inputs=inputs, outputs=outputs, name="my_model")

# model.compile(loss='mse', metrics='accuracy')
# tfjs.converters.save_keras_model(model, MDL)
# bucket().upload_model(MDL, MDL, KEY)



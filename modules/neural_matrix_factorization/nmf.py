import os
import glob
import numpy as np
from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import udf
from modules.neural_matrix_factorization.GMF.gmf import GMF
from modules.neural_matrix_factorization.MLP.mlp import MLP


class NMF(object):
    def __init__(self, gmf, mlp, spark):
        self.gmf = gmf
        self.mlp = mlp
        self.spark = spark
        
    def train(self, df, output):
        concat = tf.keras.layers.Concatenate()([self.gmf.build(df), self.mlp.build(df)])
        output = Dense(1, activation='linear')(concat)

        model = Model(inputs=[self.gmf.user_input, self.gmf.item_input, self.mlp.user_input, self.mlp.item_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(f'{output}/model/best_model_nmf.keras', save_best_only=True, monitor='val_loss', mode='min')

        model.fit(
            [self.gmf.train_user, self.gmf.train_item, self.mlp.train_user, self.mlp.train_item],
            self.gmf.train_rating,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=1,
            callbacks=[reduce_lr, model_checkpoint]
        )
        
class Predictor(object):
    def __init__(self, spark, user_col_name, item_col_name, rating_col_name):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.user_col_name_index = "INDEX_" + user_col_name
        self.item_col_name_index = "INDEX_" + item_col_name
        self.rating_col_name = rating_col_name
        self.model_gmf = None
        self.model_mlp = None
        self.model_nmf = None
        self.item_indexer = StringIndexer().setInputCol(
            self.item_col_name).setOutputCol(self.item_col_name_index).setHandleInvalid("keep")
        self.user_indexer = StringIndexer().setInputCol(
            self.user_col_name).setOutputCol(self.user_col_name_index).setHandleInvalid("keep")
        self.user_indexer_model = None
        self.item_indexer_model = None
        self.model = None
        self.spark = spark
        
    # fit all user index
    def fit_user_index(self, df, output):
        self.user_indexer_model = self.user_indexer.fit(df)
        self.user_indexer_model.write().overwrite().save(output)

    # fit all item index
    def fit_item_index(self, df, output):
        self.item_indexer_model = self.item_indexer.fit(df)
        self.item_indexer_model.write().overwrite().save(output)
        
    def fit(self, input_df, output):
        # preprocessed input data
        print("begin predict using NMF")
        encoded_df = self.user_indexer_model.transform(input_df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        encoded_df = encoded_df.orderBy('INDEX_MASV')
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        
        train_user = np.array(encoded_df.select("INDEX_MASV").collect()).flatten()
        train_item = np.array(encoded_df.select("INDEX_F_MAMH").collect()).flatten()
        train_rating = np.array(encoded_df.select("TKET").collect()).flatten()

        # Define constants
        num_users = len(train_user)
        num_items = len(train_item)
        embedding_size = 50
        self.model_gmf = GMF(num_users, num_items, embedding_size, train_user, train_item, train_rating, self.spark)
        self.model_mlp = MLP(num_users, num_items, embedding_size, train_user, train_item, train_rating, self.spark)
        
        self.model_nmf = NMF(self.model_gmf, self.model_mlp, self.spark)
        self.model_nmf.train(encoded_df, output)

def train_model(spark, csv_path, faculty):
    # loading input files - pre-processed, load all csv file
    path = f"{csv_path}/*csv"
    allcsv = glob.glob(path)
    input_file = allcsv
    
    OUTPUT_PATH = os.path.join("/app/modules/neural_matrix_factorization/output", faculty)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    USER_INDEXER_PATH = os.path.join(OUTPUT_PATH, 'user_indexer')
    os.makedirs(USER_INDEXER_PATH, exist_ok=True)
    ITEM_INDEXER_PATH = os.path.join(OUTPUT_PATH, 'item_indexer')
    os.makedirs(ITEM_INDEXER_PATH, exist_ok=True)

    # create spark session
    
    spark.sparkContext.setLogLevel("WARN")
    
    # read input files
    df = spark.read\
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file)
        
    df = df.select("MASV", "F_MAMH", "F_MAKH", "TKET", "F_DVHT")
    df = df.filter(df["F_MAKH"] == faculty)
    # print(df.count())
    df = df.withColumn("MASV", df["MASV"].cast(DoubleType()))
    df = df.withColumn("F_MAMH", df["F_MAMH"].cast(StringType()))
    df = df.withColumn("F_MAKH", df["F_MAKH"].cast(StringType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    df = df.withColumn("F_DVHT", df["F_DVHT"].cast(IntegerType()))
    predict_model = Predictor(spark, "MASV", "F_MAMH", "TKET")
    predict_model.fit_user_index(df, USER_INDEXER_PATH)
    predict_model.fit_item_index(df, ITEM_INDEXER_PATH)
    (training) = df
    df = None
    # training.show(20)

    predicted = predict_model.fit(training, OUTPUT_PATH)

    return "--------- TRAINING MODEL DONE ---------"

# if __name__ == "__main__":
#     spark = SparkSession.builder.appName("Neural Matrix Factorization")\
#             .getOrCreate()
#     faculty = "MT"        
#     train_model(spark, "/app/data/pre-processed", faculty)
#     spark.stop()
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, rand
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType
from generalized_matrix_factorization.gmf import GMF
from multi_layer_perceptron.mlp import MLP

class NMF(object):
    def __init__(self, gmf, mlp, test_df, spark):
        self.gmf = gmf
        self.mlp = mlp
        self.test_df = test_df
        self.predicted_df = None
        self.predicted_df_w_als = None
        self.rmse = None
        self.mse = None
        self.mae = None
        self.rmse_w_als = None
        self.mse_w_als = None
        self.mae_w_als = None
        self.spark = spark
        
    def train(self):
        concat = tf.keras.layers.Concatenate()([self.gmf.build(), self.mlp.build()])
        output = Dense(1, activation='linear')(concat)

        model = Model(inputs=[self.gmf.user_input, self.gmf.item_input, self.mlp.user_input, self.mlp.item_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint('../modules/neural_matrix_factorization/output/best_model_nmf.keras', save_best_only=True, monitor='val_loss', mode='min')

        model.fit(
            [self.gmf.train_user, self.gmf.train_item, self.mlp.train_user, self.mlp.train_item],
            self.gmf.train_rating,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=1,
            callbacks=[reduce_lr, model_checkpoint, early_stopping]
        )

        predictions = model.predict([self.gmf.test_user, self.gmf.test_item, self.mlp.test_user, self.mlp.test_item])
        predictions_clipped = np.clip(predictions, 0, 10)
        predicted_df = self.test_df
        predicted_pd = predicted_df.toPandas()
        predicted_df = None
        predicted_pd['prediction'] = predictions_clipped
        self.predicted_df = self.spark.createDataFrame(predicted_pd)
        
    def train_w_als(self, df):
        concat = tf.keras.layers.Concatenate()([self.mlp.build_w_als(df), self.gmf.build_w_als(df)])
        output = Dense(1, activation='linear')(concat)

        model = Model(inputs=[self.gmf.user_input, self.gmf.item_input, self.mlp.user_input, self.mlp.item_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint('../evaluation_modules/neural_matrix_factorization/output/best_model_nmf.keras', save_best_only=True, monitor='val_loss', mode='min')

        model.fit(
            [self.gmf.train_user, self.gmf.train_item, self.mlp.train_user, self.mlp.train_item],
            self.gmf.train_rating,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1,
            callbacks=[reduce_lr, model_checkpoint, early_stopping]
        )

        predictions = model.predict([self.gmf.test_user, self.gmf.test_item, self.mlp.test_user, self.mlp.test_item])
        predictions_clipped = np.clip(predictions, 0, 10)
        predicted_df = self.test_df
        predicted_pd = predicted_df.toPandas()
        predicted_df = None
        predicted_pd['prediction'] = predictions_clipped
        self.predicted_df_w_als = self.spark.createDataFrame(predicted_pd)
        
    def calc_rmse(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.rmse = evaluator.evaluate(predicted_df)
        return self.rmse
    
    def calc_mse(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mse = evaluator.evaluate(predicted_df)
        return self.mse
    
    def calc_mae(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mae",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mae = evaluator.evaluate(predicted_df)
        return self.mae
    
    def calc_rmse_w_als(self):
        predicted_df = self.predicted_df_w_als.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.rmse_w_als = evaluator.evaluate(predicted_df)
        return self.rmse_w_als
    
    def calc_mse_w_als(self):
        predicted_df = self.predicted_df_w_als.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mse_w_als = evaluator.evaluate(predicted_df)
        return self.mse_w_als
    
    def calc_mae_w_als(self):
        predicted_df = self.predicted_df_w_als.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mae",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mae_w_als = evaluator.evaluate(predicted_df)
        return self.mae_w_als

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
            self.item_col_name).setOutputCol(self.item_col_name_index)
        self.item_index_df = None
        self.user_indexer = StringIndexer().setInputCol(
            self.user_col_name).setOutputCol(self.user_col_name_index)
        self.user_index_df = None

        self.user_indexer_model = None
        self.item_indexer_model = None
        self.model = None
        self.spark = spark
        self.time = None
        
    # fit all user index
    def fit_user_index(self, df):
        self.user_indexer_model = self.user_indexer.fit(df)
        self.user_index_df = self.user_indexer_model.transform(
            df.select(self.user_col_name).distinct())

    # fit all item index
    def fit_item_index(self, df):
        self.item_indexer_model = self.item_indexer.fit(df)
        self.item_index_df = self.item_indexer_model.transform(
            df.select(self.item_col_name).distinct())
        
    def predict_using_nmf(self, input_df, predict_course_df=None):
        # preprocessed input data
        start_time = time.time()
        print("begin predict using NMF")
        encoded_df = self.user_indexer_model.transform(input_df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        encoded_df = encoded_df.orderBy('INDEX_MASV')
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        
        if predict_course_df is None:
            return
        else:
            predict_course_df = self.user_indexer_model.transform(predict_course_df)
            predict_course_df = self.item_indexer_model.transform(predict_course_df)
            predict_course_df_predict = predict_course_df.orderBy('INDEX_MASV')
        
        train_user = np.array(encoded_df.select("INDEX_MASV").collect()).flatten()
        train_item = np.array(encoded_df.select("INDEX_F_MAMH").collect()).flatten()
        train_rating = np.array(encoded_df.select("TKET").collect()).flatten()

        test_user = np.array(predict_course_df_predict.select("INDEX_MASV").collect()).flatten()
        test_item = np.array(predict_course_df_predict.select("INDEX_F_MAMH").collect()).flatten()
        test_rating = np.array(predict_course_df_predict.select("TKET").collect()).flatten()

        # Define constants
        num_users = len(train_user) + len(test_user)
        num_items = len(train_item) + len(test_item)
        embedding_size = 50
        self.model_gmf = GMF(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df_predict, self.spark)
        self.model_mlp = MLP(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df_predict, self.spark)
        # model_mlp.train()
        
        self.model_nmf = NMF(self.model_gmf, self.model_mlp, predict_course_df_predict, self.spark)
        self.model_nmf.train()
        
        end_time = time.time()
        self.time = end_time - start_time
        
        return self.model_nmf.predicted_df

    def transform(self, df):
        encoded_df = self.user_indexer_model.transform(df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        return self.model.transform(encoded_df).orderBy('INDEX_MASV')

if __name__ == "__main__":
    # loading input files - pre-processed, load all csv file
    path = "../data/pre-processed/*.csv"
    allcsv = glob.glob(path)
    input_file = allcsv
    faculty = "MT"

    # create spark session
    spark = SparkSession.builder.appName("Neural Matrix Factorization").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # read input files
    df = spark.read\
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file)
        
    print("Số lượng partition ban đầu:", df.rdd.getNumPartitions())
        
    df = df.select("MASV", "F_MAMH", "F_MAKH", "TKET", "F_DVHT")
    df = df.filter(df["F_MAKH"] == faculty)
    # print(df.count())
    df = df.withColumn("MASV", df["MASV"].cast(DoubleType()))
    df = df.withColumn("MASV", df["MASV"].cast(IntegerType()))
    df = df.withColumn("F_MAMH", df["F_MAMH"].cast(StringType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    df = df.withColumn("F_DVHT", df["F_DVHT"].cast(IntegerType()))
    predict_model = Predictor(spark, "MASV", "F_MAMH", "TKET")
    predict_model.fit_user_index(df)
    predict_model.fit_item_index(df)
    (training, test) = df.randomSplit([0.9, 0.1])
    df = None
    # training.show(20)

    predicted = predict_model.predict_using_nmf(training, test)
    # predicted.show(20)
    
    try:
        predicted.write \
            .mode("overwrite") \
            .option("header", "true") \
            .option("charset", "UTF-8") \
            .csv("../modules/neural_matrix_factorization/output")
        print("CSV file written successfully.")
    except Exception as e:
        print("Error occurred while writing CSV file: ", e)
        
    print("Root-mean-square error NMF = " + str(predict_model.model_nmf.calc_rmse()))
    print("Mean squared error NMF = " + str(predict_model.model_nmf.calc_mse()))
    print("Mean absolute error NMF = " + str(predict_model.model_nmf.calc_mae()))
    print("Time execution: " + str(predict_model.time))
    
    spark.stop()

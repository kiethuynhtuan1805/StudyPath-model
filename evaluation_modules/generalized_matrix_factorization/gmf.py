import glob
import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, ArrayType
from pyspark.sql.functions import udf, col
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

class GMF(object):
    def __init__(self, num_users, num_items, embedding_size, train_user, train_item, train_rating, test_user, test_item, test_rating, test_df, spark):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_input = Input(shape=(1,), name="user_input_gmf")
        self.item_input = Input(shape=(1,), name="item_input_gmf")
        self.train_user = train_user
        self.train_item = train_item
        self.train_rating = train_rating
        self.test_user = test_user
        self.test_item = test_item
        self.test_rating = test_rating
        self.test_df = test_df
        self.predicted_df = None
        self.als = ALS(rank=50, maxIter=15, regParam=0.01, userCol="INDEX_MASV",
                       itemCol="INDEX_F_MAMH", ratingCol="TKET", coldStartStrategy="drop")
        self.rmse = None
        self.mae = None
        self.mse = None
        self.spark = spark
    
    def normalize_features(self, df):
        # L2
        norm_expr = F.sqrt(
            sum(F.col("features")[i]**2 for i in range(len(df.first()["features"])))
        )
        
        norm_expr_safe = F.when(norm_expr == 0, 1e-6).otherwise(norm_expr)
    
        normalize_expr = F.array(*[
            F.col("features")[i] / norm_expr_safe
            for i in range(len(df.first()["features"]))
        ])
    
        normalized_df = df.withColumn("normalized_features", normalize_expr)
    
        return normalized_df
    
    def build(self):   
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size, name="user_embedding_gmf", trainable=True)(self.user_input)
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_size, name="item_embedding_gmf", trainable=True)(self.item_input)

        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
        
        vector = Multiply()([user_vec, item_vec])
        
        return vector
    
    def build_w_als(self, df):   
        als_model = self.als.fit(df)
        
        normalized_user_df = self.normalize_features(als_model.userFactors).withColumnRenamed("id", "INDEX_MASV")
        normalized_item_df = self.normalize_features(als_model.itemFactors).withColumnRenamed("id", "INDEX_F_MAMH")
    
        als_user_weights_df = df.join(normalized_user_df, on="INDEX_MASV", how="left")
        als_item_weights_df = df.join(normalized_item_df, on="INDEX_F_MAMH", how="left")
    
        normalized_user_embeddings = np.array(als_user_weights_df.select("normalized_features").rdd.map(lambda x: x[0]).collect())
        normalized_item_embeddings = np.array(als_item_weights_df.select("normalized_features").rdd.map(lambda x: x[0]).collect())
    
        # Keras Embeddings
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size, name="user_embedding_gmf", weights=[normalized_user_embeddings])(self.user_input)
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_size, name="item_embedding_gmf", weights=[normalized_item_embeddings])(self.item_input)
    
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
    
        vector = Multiply()([user_vec, item_vec])
    
        return vector
    
    def train(self):
        dense1 = Dense(128, activation='relu')(self.build())
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(1, activation='linear')(dense2)
        model = Model(inputs=[self.user_input, self.item_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint('../evaluation_modules/neural_matrix_factorization/output/best_model_gmf.keras', save_best_only=True, monitor='val_loss', mode='min')
        
        history = model.fit(
            [self.train_user, self.train_item],
            self.train_rating,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stopping, reduce_lr, model_checkpoint]
        )
        
        predictions = model.predict([self.test_user, self.test_item])
        predictions_clipped = np.clip(predictions, 0, 10)
        predicted_df = self.test_df
        predicted_pd = predicted_df.toPandas()
        predicted_df = None
        predicted_pd['prediction'] = predictions_clipped
        self.predicted_df = self.spark.createDataFrame(predicted_pd)
    
    def calc_rmse(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.rmse = evaluator.evaluate(predicted_df)
        return self.rmse
    
    def calc_mae(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mae",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mae = evaluator.evaluate(predicted_df)
        return self.mae
    
    def calc_mse(self):
        predicted_df = self.predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(
            metricName="mse",
            labelCol="TKET",
            predictionCol="prediction"
        )
        self.mse = evaluator.evaluate(predicted_df)
        return self.mse
        
class Predictor(object):
    def __init__(self, spark, user_col_name, item_col_name, rating_col_name):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.user_col_name_index = "INDEX_" + user_col_name
        self.item_col_name_index = "INDEX_" + item_col_name
        self.rating_col_name = rating_col_name
        self.model_gmf = None
        
        self.item_indexer = StringIndexer().setInputCol(
            self.item_col_name).setOutputCol(self.item_col_name_index)
        self.item_index_df = None
        self.user_indexer = StringIndexer().setInputCol(
            self.user_col_name).setOutputCol(self.user_col_name_index)
        self.user_index_df = None

        self.user_indexer_model = None
        self.item_indexer_model = None
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
        
    def predict_using_gmf(self, input_df, predict_course_df=None):
        # preprocessed input data
        start_time = time.time()
        print("begin predict using GMF")
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
        num_users = len(train_user)
        num_items = len(train_item)
        embedding_size = 20
        self.model_gmf = GMF(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df_predict, self.spark)
        self.model_gmf.train()
        
        end_time = time.time()
        self.time = end_time - start_time
        
        return self.model_gmf.predicted_df
        
if __name__ == "__main__":
    # loading input files - pre-processed, load all csv file
    path = "../data/pre-processed/*.csv"
    allcsv = glob.glob(path)
    input_file = allcsv
    faculty = "MT"

    # create spark session
    spark = SparkSession.builder.appName("Generalized Matrix Factorization").getOrCreate()
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
    training.show(20)

    predicted = predict_model.predict_using_gmf(training, test)
    predicted.show(20)
    
    print("Root-mean-square error of GMF model = " + str(predict_model.model_gmf.calc_rmse()))
    print("Mean squared error of GMF model = " + str(predict_model.model_gmf.calc_mse()))
    print("Mean absolute error of GMF model = " + str(predict_model.model_gmf.calc_mae()))
    print("Time execution: " + str(predict_model.time))
    
    spark.stop()
import glob
import numpy as np
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from generalized_matrix_factorization.gmf import GMF
from multi_layer_perceptron.mlp import MLP
from neural_matrix_factorization.nmf import NMF

class Predictor(object):

    def __init__(self, spark, user_col_name, item_col_name, rating_col_name):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.user_col_name_index = "INDEX_" + user_col_name
        self.item_col_name_index = "INDEX_" + item_col_name
        self.rating_col_name = rating_col_name
        self.als = ALS(rank=50, maxIter=15, regParam=0.01, userCol=self.user_col_name_index,
                       itemCol=self.item_col_name_index, ratingCol=rating_col_name, coldStartStrategy="drop")
        self.model_gmf = None
        self.model_mlp = None
        self.model_nmf = None
        self.model_gmf_w_als = None
        self.model_mlp_w_als = None
        self.model_nmf_w_als = None
        self.item_indexer = StringIndexer().setInputCol(
            self.item_col_name).setOutputCol(self.item_col_name_index)
        self.item_index_df = None
        self.user_indexer = StringIndexer().setInputCol(
            self.user_col_name).setOutputCol(self.user_col_name_index)
        self.user_index_df = None

        self.user_indexer_model = None
        self.item_indexer_model = None
        self.model = None
        self.item_similarity = None
        self.user_similarity = None
        self.spark = spark
        
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

    def fit(self, training_df):
        encoded_df = self.item_indexer_model.transform(training_df)
        encoded_df = self.user_indexer_model.transform(encoded_df)
        self.model = self.als.fit(encoded_df)

    def calc_ibcf(self):
        item_factor = self.model.itemFactors
        item_factor.createOrReplaceTempView("ItemFactor")

        # function to calculate cosine similarity between two array
        def cosine_similarity(item1, item2):
            np_array_item1 = np.array(item1)
            np_array_item2 = np.array(item2)
            dot_product = np.linalg.norm(
                np_array_item1) * np.linalg.norm(np_array_item2)
            if dot_product == 0:
                return 0.0
            return float(np.dot(np_array_item1, np_array_item2) / dot_product)
        cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        item_similarity = self.spark.sql(
            "SELECT I1.id as id1, I2.id as id2, I1.features as features1, I2.features as features2  FROM ItemFactor I1, ItemFactor I2 WHERE I1.id != I2.id")
        self.item_similarity = item_similarity.withColumn("similarity",
                                                          cosine_similarity_udf(item_similarity["features1"],
                                                                                item_similarity["features2"]))
        self.item_similarity.show(20)
        

    # input_df will have 1 student id and all course that the student already studied
    # first we will index all the course the student already studied and normalize all score
    # then map similarity data to the already studied course
    # then check if predict_course_df is None or not, if it None, then predict all the remaining course,
    # if not transform the predict_course_df to get the index of predict course
    # then begin predict function (use first 5 relevant course to that course that the student already studied)
    def predict_using_cosine_similarity_ibcf(self, input_df, predict_course_df=None):
        # preprocessed input data
        print("begin predict using cosine similarity for IBCF")
        encoded_df = self.item_indexer_model.transform(input_df)
        encoded_df = self.user_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))

        # get predict course df (remaining course)
        if predict_course_df is None:
            predict_course_df_predict = encoded_df.join(self.item_index_df,
                                                        encoded_df[self.item_col_name_index] != self.item_index_df[
                                                            self.item_col_name_index]) \
                .select(self.user_col_name, self.item_col_name_index)
        else:
            predict_course_df = self.item_indexer_model.transform(predict_course_df)
            predict_course_df_predict = predict_course_df.drop(
                self.rating_col_name)

        # get all value that can participate in evaluate final score
        similarity_score_df = encoded_df.join(self.item_similarity,
                                              encoded_df[self.item_col_name_index] == self.item_similarity['id1']) \
            .select(self.user_col_name, self.rating_col_name, 'id1', 'id2', 'similarity') \


        def predict(list_score, list_similarity):
            sum_simi = sum(list_similarity)
            if sum_simi == 0:
                return 0.0
            return sum([list_score[i] * list_similarity[i] for i in range(len(list_score))]) / sum(list_similarity)

        predict_udf = udf(predict, DoubleType())
        window = Window.partitionBy([spark_func.col(self.user_col_name), spark_func.col(
            self.item_col_name_index)]).orderBy(spark_func.col('similarity').desc())

        predict_course_df_predict = predict_course_df_predict.join(similarity_score_df.withColumnRenamed("id2", self.item_col_name_index),
                                                                   [self.item_col_name_index, self.user_col_name])\
            .select("*", spark_func.rank().over(window).alias("rank"))\
            .filter(spark_func.col("rank") <= 5).groupby(self.user_col_name, self.item_col_name_index)\
            .agg(spark_func.collect_list(self.rating_col_name).alias("list_score"), spark_func.collect_list("similarity").alias("list_similarity"))
        predict_course_df_predict = predict_course_df_predict.withColumn(
            "prediction", predict_udf(spark_func.col("list_score"), spark_func.col("list_similarity")))

        if predict_course_df is not None and self.rating_col_name in predict_course_df.columns:
            predict_course_df_predict = predict_course_df_predict.join(
                predict_course_df, [self.user_col_name, self.item_col_name_index])

        return predict_course_df_predict
    
    def calc_ubcf(self):
        user_factor = self.model.userFactors
        user_factor.createOrReplaceTempView("UserFactor")

        # function to calculate cosine similarity between two array
        def cosine_similarity(user1, user2):
            np_array_user1 = np.array(user1)
            np_array_user2 = np.array(user2)
            dot_product = np.linalg.norm(
                np_array_user1) * np.linalg.norm(np_array_user2)
            if dot_product == 0:
                return 0.0
            return float(np.dot(np_array_user1, np_array_user2) / dot_product)
        cosine_similarity_udf = udf(cosine_similarity, DoubleType())
        user_similarity = self.spark.sql(
            "SELECT U1.id as id1, U2.id as id2, U1.features as features1, U2.features as features2  FROM UserFactor U1, UserFactor U2 WHERE U1.id != U2.id")
        self.user_similarity = user_similarity.withColumn("similarity",
                                                          cosine_similarity_udf(user_similarity["features1"],
                                                                                user_similarity["features2"]))
        self.user_similarity.show(20)
    
    def predict_using_cosine_similarity_ubcf(self, input_df, predict_course_df=None):
        # preprocessed input data
        print("begin predict using cosine similarity for UBCF")
        encoded_df = self.user_indexer_model.transform(input_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))

        # Determine which user to predict
        if predict_course_df is None:
            predict_course_df_predict = encoded_df.join(self.user_index_df,
                                                        encoded_df[self.user_col_name_index] != self.user_index_df[
                                                            self.user_col_name_index]) \
                .select(self.user_col_name, self.user_col_name_index)
        else:
            predict_course_df = self.user_indexer_model.transform(predict_course_df)
            predict_course_df_predict = predict_course_df.drop(
                self.rating_col_name)

        # Get all values that can participate in evaluating the final score
        similarity_score_df = encoded_df.join(self.user_similarity,
                                              encoded_df[self.user_col_name_index] == self.user_similarity['id1']) \
            .select(self.item_col_name, self.rating_col_name, 'id1', 'id2', 'similarity')

        def predict(list_score, list_similarity):
            sum_simu = sum(list_similarity)
            if sum_simu == 0:
                return 0.0
            return sum([list_score[i] * list_similarity[i] for i in range(len(list_score))]) / sum(list_similarity)

        predict_udf = udf(predict, DoubleType())
        window = Window.partitionBy([spark_func.col(self.item_col_name), spark_func.col(
            self.user_col_name_index)]).orderBy(spark_func.col('similarity').desc())

        predict_course_df_predict = predict_course_df_predict.join(similarity_score_df.withColumnRenamed("id2", self.user_col_name_index),
                                                               [self.user_col_name_index, self.item_col_name]) \
            .select("*", spark_func.rank().over(window).alias("rank")) \
            .filter(spark_func.col("rank") <= 5).groupby(self.item_col_name, self.user_col_name_index) \
            .agg(spark_func.collect_list(self.rating_col_name).alias("list_score"), spark_func.collect_list("similarity").alias("list_similarity"))
        predict_course_df_predict = predict_course_df_predict.withColumn(
            "prediction", predict_udf(spark_func.col("list_score"), spark_func.col("list_similarity")))

        if predict_course_df is not None and self.rating_col_name in predict_course_df.columns:
            predict_course_df_predict = predict_course_df_predict.join(
                predict_course_df, [self.item_col_name, self.user_col_name_index])

        return predict_course_df_predict
        
    def predict_using_nmf(self, input_df, predict_course_df=None):
        # preprocessed input data
        print("begin predict using NMF")
        encoded_df = self.user_indexer_model.transform(input_df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        
        if predict_course_df is None:
            return
        else:
            predict_course_df = self.user_indexer_model.transform(predict_course_df)
            predict_course_df = self.item_indexer_model.transform(predict_course_df)
        
        train_user = np.array(encoded_df.select("INDEX_MASV").collect()).flatten()
        train_item = np.array(encoded_df.select("INDEX_F_MAMH").collect()).flatten()
        train_rating = np.array(encoded_df.select("TKET").collect()).flatten()

        test_user = np.array(predict_course_df.select("INDEX_MASV").collect()).flatten()
        test_item = np.array(predict_course_df.select("INDEX_F_MAMH").collect()).flatten()
        test_rating = np.array(predict_course_df.select("TKET").collect()).flatten()

        # Define constants
        num_users = len(train_user)
        num_items = len(train_item)
        embedding_size = 50
        self.model_gmf = GMF(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df, self.spark)
        self.model_mlp = MLP(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df, self.spark)
        
        
        self.model_gmf.train()
        self.model_gmf.predicted_df.show(20)
        
        self.model_mlp.train()
        self.model_mlp.predicted_df.show(20)
        
        self.model_nmf = NMF(self.model_gmf, self.model_mlp, predict_course_df, self.spark)
        self.model_nmf.train()
        
        return self.model_nmf.predicted_df
    
    def predict_using_nmf_w_als(self, input_df, predict_course_df=None):
        # preprocessed input data
        print("begin predict using NMF and ALS be weights")
        encoded_df = self.user_indexer_model.transform(input_df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        
        if predict_course_df is None:
            return
        else:
            predict_course_df = self.user_indexer_model.transform(predict_course_df)
            predict_course_df = self.item_indexer_model.transform(predict_course_df)
        
        train_user = np.array(encoded_df.select("INDEX_MASV").collect()).flatten()
        train_item = np.array(encoded_df.select("INDEX_F_MAMH").collect()).flatten()
        train_rating = np.array(encoded_df.select("TKET").collect()).flatten()

        test_user = np.array(predict_course_df.select("INDEX_MASV").collect()).flatten()
        test_item = np.array(predict_course_df.select("INDEX_F_MAMH").collect()).flatten()
        test_rating = np.array(predict_course_df.select("TKET").collect()).flatten()

        # Define constants
        num_users = len(train_user)
        num_items = len(train_item)
        embedding_size = 50
        self.model_gmf_w_als = GMF(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df, self.spark)
        self.model_mlp_w_als = MLP(num_users, num_items, embedding_size, train_user, train_item, train_rating,
                        test_user, test_item, test_rating, predict_course_df, self.spark)
        
        self.model_nmf_w_als = NMF(self.model_gmf_w_als, self.model_mlp_w_als, predict_course_df, self.spark)
        self.model_nmf_w_als.train_w_als(encoded_df)
        
        return self.model_nmf_w_als.predicted_df_w_als
    
    def transform(self, df):
        encoded_df = self.user_indexer_model.transform(df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        return self.model.transform(encoded_df)

    def calc_rmse(self, predicted_df):
        predicted_df = predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)
    
    def calc_mse(self, predicted_df):
        predicted_df = predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)
    
    def calc_mae(self, predicted_df):
        predicted_df = predicted_df.select("TKET", "prediction")
        
        evaluator = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)
    

if __name__ == "__main__":
    # loading input files - pre-processed, load all csv file
    path = "../data/pre-processed/*.csv"
    allcsv = glob.glob(path)
    input_file = allcsv
    faculty = "VL"

    # create spark session
    spark = SparkSession.builder.appName("Student Performance Prediction")\
                            .config("spark.driver.extraJavaOptions", "-Xss4m") \
                            .config("spark.executor.extraJavaOptions", "-Xss4m") \
                            .getOrCreate()
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
    df = df.withColumn("F_MAMH", df["F_MAMH"].cast(StringType()))
    df = df.withColumn("F_MAKH", df["F_MAKH"].cast(StringType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    df = df.withColumn("F_DVHT", df["F_DVHT"].cast(IntegerType()))
    predict_model = Predictor(spark, "MASV", "F_MAMH", "TKET")
    predict_model.fit_user_index(df)
    predict_model.fit_item_index(df)
    (training, test) = df.randomSplit([0.9, 0.1], seed=42)
    df = None
    training.show(20)
    
    predict_model.fit(training)
    
    # normal prediction using als
    predicted = predict_model.transform(test)
    print("SHOW ALS")
    predicted.show(20)
    als_rmse = predict_model.calc_rmse(predicted)
    als_mse = predict_model.calc_mse(predicted)
    als_mae = predict_model.calc_mae(predicted)
    
    # predict_model.calc_ibcf()
    predicted = predict_model.predict_using_cosine_similarity_ibcf(training, test)
    print("SHOW IBCF")
    predicted.show(20)
    ibcf_rmse = predict_model.calc_rmse(predicted)
    ibcf_mse = predict_model.calc_mse(predicted)
    ibcf_mae = predict_model.calc_mae(predicted)
    
    # predict_model.calc_ubcf()
    # predicted = predict_model.predict_using_cosine_similarity_ubcf(training, test)
    # predicted.show(20)
    # ubcf_rmse = predict_model.calc_rmse(predicted)
    # ubcf_mse = predict_model.calc_mse(predicted)
    # ubcf_mae = predict_model.calc_mae(predicted)

    predicted = predict_model.predict_using_nmf(training, test)
    print("SHOW NMF")
    predicted.show(20)
    
    predicted = predict_model.predict_using_nmf_w_als(training, test)
    print("SHOW NMF W ALS")
    predicted.show(20)
    
    print("----------------------------")
    print("Root-mean-square error ALS = " + str(als_rmse))
    print("Root-mean-square error IBCF = " + str(ibcf_rmse))
    # print("Root-mean-square error UBCF = " + str(ubcf_rmse))
    print("Root-mean-square error GMF = " + str(predict_model.model_gmf.calc_rmse()))
    print("Root-mean-square error MLP = " + str(predict_model.model_mlp.calc_rmse()))
    print("Root-mean-square error NMF = " + str(predict_model.model_nmf.calc_rmse()))
    print("Root-mean-square error NMF with ALS = " + str(predict_model.model_nmf_w_als.calc_rmse_w_als()))
    print("----------------------------")
    print("Mean squared error ALS = " + str(als_mse))
    print("Mean squared error IBCF = " + str(ibcf_mse))
    # print("Mean squared error UBCF = " + str(ubcf_mse))
    print("Mean squared error GMF = " + str(predict_model.model_gmf.calc_mse()))
    print("Mean squared error MLP = " + str(predict_model.model_mlp.calc_mse()))
    print("Mean squared error NMF = " + str(predict_model.model_nmf.calc_mse()))
    print("Mean squared error NMF with ALS = " + str(predict_model.model_nmf_w_als.calc_mse_w_als()))
    print("----------------------------")
    print("Mean absolute error ALS = " + str(als_mae))
    print("Mean absolute error IBCF = " + str(ibcf_mae))
    # print("Mean absolute error UBCF = " + str(ubcf_mae))
    print("Mean absolute error GMF = " + str(predict_model.model_gmf.calc_mae()))
    print("Mean absolute error MLP = " + str(predict_model.model_mlp.calc_mae()))
    print("Mean absolute error NMF = " + str(predict_model.model_nmf.calc_mae()))
    print("Mean absolute error NMF with ALS = " + str(predict_model.model_nmf_w_als.calc_mae_w_als()))
    # predicted_csv = predicted.withColumn(
    #     "list_score_ibcf", concat_ws(",", "list_score"))
    # predicted_csv = predicted_csv.withColumn(
    #     "list_similarity_ibcf", concat_ws(",", "list_similarity"))
    # try:
    #     predicted_csv.write \
    #         .mode("overwrite") \
    #         .option("header", "true") \
    #         .option("charset", "UTF-8") \
    #         .csv("../modules/deep-neural-network/output")
    #     print("CSV file written successfully.")
    # except Exception as e:
    #     print("Error occurred while writing CSV file: ", e)
    
    spark.stop()

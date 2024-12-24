import glob
import numpy as np
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession

class Predictor(object):

    def __init__(self, spark, user_col_name, item_col_name, rating_col_name):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.user_col_name_index = "INDEX_" + user_col_name
        self.item_col_name_index = "INDEX_" + item_col_name
        self.rating_col_name = rating_col_name
        self.als = ALS(rank=30, maxIter=15, regParam=0.01, userCol=self.user_col_name_index,
                       itemCol=self.item_col_name_index, ratingCol=rating_col_name, coldStartStrategy="drop")
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
        encoded_df = encoded_df.orderBy('INDEX_MASV')
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        self.model = self.als.fit(encoded_df)

    def calc_ibcf(self):
        self.model.itemFactors.show()
        item_factor = self.model.itemFactors.orderBy("id")
        self.model.itemFactors.show()
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
    # then begin predict function (use first 5 relevant course to that course that the student already studied).
    
    def predict_using_cosine_similarity_ibcf(self, input_df, predict_course_df=None):
        # preprocessed input data
        print("begin predict using cosine similarity for IBCF")
        encoded_df = self.item_indexer_model.transform(input_df)
        encoded_df = self.user_indexer_model.transform(encoded_df)
        encoded_df = encoded_df.orderBy('INDEX_MASV')
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

        return predict_course_df_predict.orderBy("MASV")
    
    def calc_ubcf(self):
        user_factor = self.model.userFactors.orderBy("id")
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
        encoded_df = encoded_df.orderBy('INDEX_MASV')
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

        return predict_course_df_predict.orderBy("MASV")

    def transform(self, df):
        encoded_df = self.user_indexer_model.transform(df)
        encoded_df = self.item_indexer_model.transform(encoded_df)
        normalize_rating_udf = udf(
            lambda p: 0.0 if p > 10 else p, DoubleType())
        encoded_df = encoded_df.withColumn(
            self.rating_col_name, normalize_rating_udf(encoded_df[self.rating_col_name]))
        return self.model.transform(encoded_df).orderBy('INDEX_MASV')

    def calc_rmse(self, predicted_df):
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)
    
    def calc_mse(self, predicted_df):
        evaluator = RegressionEvaluator(metricName="mse", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)
    
    def calc_mae(self, predicted_df):
        evaluator = RegressionEvaluator(metricName="mae", labelCol="TKET",
                                    predictionCol="prediction")
        
        return evaluator.evaluate(predicted_df)

if __name__ == "__main__":
    # loading input files - pre-processed, load all csv file
    path = "../data/pre-processed/*.csv"
    allcsv = glob.glob(path)
    input_file = allcsv
    faculty = "MT"

    # create spark session
    spark = SparkSession.builder.appName("Student Performance Prediction").getOrCreate()
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
    
    predict_model.fit(training)
    
    predict_model.als.save('./output')
    
    # normal prediction using als
    predicted = predict_model.transform(test)
    predicted.show(20)
    als_rmse = predict_model.calc_rmse(predicted)
    als_mse = predict_model.calc_mse(predicted)
    als_mae = predict_model.calc_mae(predicted)
    
    
    predict_model.calc_ibcf()
    predicted = predict_model.predict_using_cosine_similarity_ibcf(training, test)
    predicted.show(20)
    ibcf_rmse = predict_model.calc_rmse(predicted)
    ibcf_mse = predict_model.calc_mse(predicted)
    ibcf_mae = predict_model.calc_mae(predicted)
    
    predict_model.calc_ubcf()
    predicted = predict_model.predict_using_cosine_similarity_ubcf(training, test)
    predicted.show(20)
    ubcf_rmse = predict_model.calc_rmse(predicted)
    ubcf_mse = predict_model.calc_mse(predicted)
    ubcf_mae = predict_model.calc_mae(predicted)
    
    print("Root-mean-square error ALS = " + str(als_rmse))
    print("Root-mean-square error IBCF = " + str(ibcf_rmse))
    print("Root-mean-square error UBCF = " + str(ubcf_rmse))
    print("----------------------------")
    print("Mean squared error ALS = " + str(als_mse))
    print("Mean squared error IBCF = " + str(ibcf_mse))
    print("Mean squared error UBCF = " + str(ubcf_mse))
    print("----------------------------")
    print("Mean absolute error ALS = " + str(als_mae))
    print("Mean absolute error IBCF = " + str(ibcf_mae))
    print("Mean absolute error UBCF = " + str(ubcf_mae))
    
    spark.stop()

import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pyspark.ml.feature import  StringIndexerModel
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, desc, when, col, sum
from itertools import chain
from pyspark.sql import SparkSession
from modules.recommender_module.rcm import RCM

class NMF(object):
    def __init__(self, train_user, train_item, train_rating, test_user, test_item, test_df, spark):
        self.train_user = train_user
        self.train_item = train_item
        self.train_rating = train_rating
        self.test_user = test_user
        self.test_item = test_item
        self.test_df = test_df
        self.predicted_df = None
        self.spark = spark
        self.model = None
        
    def update_embedding(self):
        embedding_layer_mlp = self.model.get_layer("user_embedding_mlp")
        embedding_layer_gmf = self.model.get_layer("user_embedding_gmf")
    
        embedding_size = embedding_layer_mlp.output_dim
        n_users_old = embedding_layer_mlp.input_dim
        n_users_new = n_users_old + len(self.train_user)

        new_mlp_user_embedding = Embedding(input_dim=n_users_new, output_dim=embedding_size, name="user_embedding_mlp", trainable=True)
        new_gmf_user_embedding = Embedding(input_dim=n_users_new, output_dim=embedding_size, name="user_embedding_gmf", trainable=True)

        new_mlp_user_embedding.build((None,))
        new_gmf_user_embedding.build((None,))

        new_mlp_embeddings = tf.concat([
            embedding_layer_mlp.get_weights()[0],
            tf.random.normal((len(self.train_user), embedding_size))
        ], axis=0)

        new_gmf_embeddings = tf.concat([
            embedding_layer_gmf.get_weights()[0],
            tf.random.normal((len(self.train_user), embedding_size))
        ], axis=0)
        
        new_mlp_user_embedding.embeddings.assign(new_mlp_embeddings)
        new_gmf_user_embedding.embeddings.assign(new_gmf_embeddings)
        
        embedding_layer_mlp.input_dim = n_users_new
        embedding_layer_gmf.input_dim = n_users_new
        
        self.model._layers[self.model.layers.index(embedding_layer_mlp)] = new_mlp_user_embedding
        self.model._layers[self.model.layers.index(embedding_layer_gmf)] = new_gmf_user_embedding
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def predict(self, faculty):
        # Load model
        self.model = load_model(f'/app/modules/neural_matrix_factorization/output/{faculty}/model/best_model_nmf.keras')
        
        self.update_embedding()
        
        # Fine-tune
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)
        
        self.model.fit(
            [self.train_user, self.train_item, self.train_user, self.train_item],
            self.train_rating,
            epochs=15,
            batch_size=1,
            validation_split=0.3,
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )
        
        predictions = self.model.predict([self.test_user, self.test_item, self.test_user, self.test_item])
        predictions_clipped = np.clip(predictions, 0, 10)
        predicted_df = self.test_df
        predicted_pd = predicted_df.toPandas()
        predicted_df = None
        predicted_pd['prediction'] = predictions_clipped
        self.predicted_df = self.spark.createDataFrame(predicted_pd)
        
class Predictor(object):
    def __init__(self, spark, user_col_name, item_col_name, rating_col_name, faculty):
        self.user_col_name = user_col_name
        self.item_col_name = item_col_name
        self.loaded_user_indexer_model = StringIndexerModel.load(f"/app/modules/neural_matrix_factorization/output/{faculty}/user_indexer")
        self.loaded_item_indexer_model = StringIndexerModel.load(f"/app/modules/neural_matrix_factorization/output/{faculty}/item_indexer")
        self.rating_col_name = rating_col_name
        self.model_nmf = None
        self.spark = spark
        self.faculty = faculty
        
    def fit_item_index(self, df):
        distinct_animals = [row[self.item_col_name] for row in df.select(self.item_col_name).distinct().collect()]
        existing_labels = self.loaded_item_indexer_model.labels

        new_labels = [animal for animal in distinct_animals if animal not in existing_labels]
        all_labels = existing_labels + new_labels

        animal_index_map = {animal: idx for idx, animal in enumerate(all_labels)}
        mapping_expr = F.create_map([F.lit(x) for x in chain(*animal_index_map.items())])

        indexed_new_data = df.withColumn("INDEX_F_MAMH", mapping_expr.getItem(F.col(self.item_col_name)).cast(IntegerType()))

        return indexed_new_data
        
    def predict_using_nmf(self, input_df , predict_course_df=None):
        # preprocessed input data
        print("begin predict using NMF")
        encoded_df = self.loaded_user_indexer_model.transform(input_df)
        encoded_df = self.fit_item_index(encoded_df)
        
        train_user = np.array(encoded_df.select("INDEX_MASV").collect()).flatten()
        train_item = np.array(encoded_df.select("INDEX_F_MAMH").collect()).flatten()
        train_rating = np.array(encoded_df.select("TKET").collect()).flatten()
        
        if predict_course_df is None:
            return
        else:
            predict_course_df = self.loaded_user_indexer_model.transform(predict_course_df)
            predict_course_df = self.fit_item_index(predict_course_df)

        test_user = np.array(predict_course_df.select("INDEX_MASV").collect()).flatten()
        test_item = np.array(predict_course_df.select("INDEX_F_MAMH").collect()).flatten()
        
        self.model_nmf = NMF(train_user, train_item, train_rating, test_user, test_item, predict_course_df, self.spark)
        self.model_nmf.predict(self.faculty)
        
        return self.model_nmf.predicted_df

def main(spark, path_course, faculty, specialize, subSpecialize, semester, masv, student_grade, credits):
    path_course = f"{path_course}/{faculty}/*.csv"
    allcsv_course = glob.glob(path_course)
    input_file_course = allcsv_course

    spark.sparkContext.setLogLevel("WARN")

    predict_model = Predictor(spark, "MASV", "F_MAMH", "TKET", faculty)
    
    # Student Dataframe
    student_df = spark.createDataFrame(student_grade)
    
    student_df_2 = student_df.select("F_MAMH", "F_TENMHVN", "F_DVHT", "TKET") \
            .withColumn("TKET", when(col("TKET").isin("CT", "VT", "KD", "RT", "CD"), "0") \
            .when(col("TKET") == "D", "10") \
            .when(col("TKET").cast("float").isNotNull(), 
                when(col("TKET").cast("float") > 10.0, "0") 
                .otherwise(col("TKET"))) \
            .otherwise("REMOVE"))
            
    student_df_2.createOrReplaceTempView("data")

    student_df_3 = spark.sql("SELECT * FROM data WHERE TKET <> 'REMOVE'")
    
    student_df_4 = student_df_3.withColumn("MASV", lit(masv).cast(DoubleType())) \
        .withColumn("F_TENMHVN", student_df_3["F_TENMHVN"].cast(StringType())) \
        .withColumn("F_MAMH", student_df_3["F_MAMH"].cast(StringType())) \
        .withColumn("F_DVHT", student_df_3["F_DVHT"].cast(IntegerType())) \
        .withColumn("TKET", student_df_3["TKET"].cast(DoubleType())) 
        
    student_df_4.show(20)
        
    student_dvht = student_df_4.select(sum("F_DVHT")).collect()[0][0]
    
    # Course Dataframe
    course_df = spark.read\
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file_course)
        
    course_df_2 = course_df.select("NAMHOC", "F_MAMH", "F_TENMHVN", "F_MAKH", "F_MANG", "F_DVHT", "KHOI", "COMPULSARY", "LEVEL")
    
    course_df_2.createOrReplaceTempView("data")

    course_df_3 = spark.sql(f"SELECT * FROM data WHERE (NAMHOC = '{2020}') and (F_MAKH = '{faculty}' or F_MAKH = '{"ALL"}') and (F_MANG = '{specialize}' or F_MANG = '{subSpecialize}' or F_MANG = '{"ALL"}')")
    
    course_df_4 = course_df_3.withColumn("F_MAMH", course_df_3["F_MAMH"].cast(StringType())) \
        .withColumn("MASV", lit(masv).cast(DoubleType())) \
        .withColumn("F_MAKH", course_df_3["F_MAKH"].cast(StringType())) \
        .withColumn("F_TENMHVN", course_df_3["F_TENMHVN"].cast(StringType())) \
        .withColumn("F_MANG", course_df_3["F_MANG"].cast(StringType())) \
        .withColumn("F_DVHT", course_df_3["F_DVHT"].cast(IntegerType())) \
        .withColumn("KHOI", course_df_3["KHOI"].cast(StringType())) \
        .withColumn("COMPULSARY", course_df_3["COMPULSARY"].cast(IntegerType())) \
        .withColumn("LEVEL", course_df_3["LEVEL"].cast(IntegerType()))
        
    course_df_4.show(20)
    
    unlearned_courses_df = course_df_4.join(student_df_4, 
                                        on=["F_MAMH", "F_DVHT"], 
                                        how="left_anti")
    unlearned_courses_df.show(20)
    
    filtered_df = unlearned_courses_df.filter(unlearned_courses_df["COMPULSARY"] != 0)
    
    remain_dvht = filtered_df.select(sum("F_DVHT")).collect()[0][0]
    
    predicted = predict_model.predict_using_nmf(student_df_4, unlearned_courses_df)
    
    unique_values = predicted.select("COMPULSARY").distinct().orderBy("COMPULSARY").rdd.flatMap(lambda x: x).collect()
    
    array = []

    for value in unique_values:
        filtered_df = predicted.filter(predicted.COMPULSARY == value).orderBy(desc("prediction"))
        result = filtered_df.select("F_MAMH", "F_TENMHVN", "F_DVHT", "COMPULSARY", "KHOI", "LEVEL", "prediction").collect()
        values = [row.asDict() for row in result]
        array.append(values)
        
    # for i in array:
    #     i.show()
    
    tctd = credits - remain_dvht - student_dvht
    
    rcm = RCM(array)
    
    result = rcm.recommend(18, semester, tctd, 1, 1, 5, 1) 
    
    return array[0]

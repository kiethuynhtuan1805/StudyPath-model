import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

class MLP(object):
    def __init__(self, num_users, num_items, embedding_size, train_user, train_item, train_rating, spark):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.als = ALS(rank=embedding_size, maxIter=15, regParam=0.01, userCol="INDEX_MASV",
                       itemCol="INDEX_F_MAMH", ratingCol="TKET", coldStartStrategy="drop")
        self.user_input = Input(shape=(1,), name="user_input_mlp")
        self.item_input = Input(shape=(1,), name="item_input_mlp")
        self.train_user = train_user
        self.train_item = train_item
        self.train_rating = train_rating
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
        
    def build(self, df):
        als_model = self.als.fit(df)
        
        normalized_user_df = self.normalize_features(als_model.userFactors).withColumnRenamed("id", "INDEX_MASV")
        normalized_item_df = self.normalize_features(als_model.itemFactors).withColumnRenamed("id", "INDEX_F_MAMH")
    
        als_user_weights_df = df.join(normalized_user_df, on="INDEX_MASV", how="left")
        als_item_weights_df = df.join(normalized_item_df, on="INDEX_F_MAMH", how="left")
    
        normalized_user_embeddings = np.array(als_user_weights_df.select("normalized_features").rdd.map(lambda x: x[0]).collect())
        normalized_item_embeddings = np.array(als_item_weights_df.select("normalized_features").rdd.map(lambda x: x[0]).collect())
    
        # Keras Embeddings
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size, name="user_embedding_mlp", weights=[normalized_user_embeddings], trainable=True)(self.user_input)
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_size, name="item_embedding_mlp", weights=[normalized_item_embeddings], trainable=True)(self.item_input)
    
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
    
        concat = Concatenate()([user_vec, item_vec])
        dense1 = Dense(256, activation='relu')(concat)
        dense3 = Dense(64, activation='relu')(dense1)
        output = Dense(32, activation='relu')(dense3)
    
        return output
from __future__ import print_function
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DecimalType, DoubleType

def read_course(spark, csv_path, input_file_course, output_path, faculty):
    # loading input files - pre-processed, load all csv file
    path = f"{csv_path}/*.csv"
    allcsv = glob.glob(path)
    input_file = allcsv

    # spark application
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
    df = df.withColumn("MASV", df["MASV"].cast(DoubleType()))
    df = df.withColumn("F_MAMH", df["F_MAMH"].cast(StringType()))
    df = df.withColumn("F_MAKH", df["F_MAKH"].cast(StringType()))
    df = df.withColumn("TKET", df["TKET"].cast(DoubleType()))
    df = df.withColumn("F_DVHT", df["F_DVHT"].cast(IntegerType()))
    
    # load original data courses
    course_df = spark.read.format("com.crealytics.spark.excel")\
        .option("useHeader", "true") \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("maxRowsInMemory", 2000) \
        .option("inferSchema", "false") \
        .option("addColorColumns", "False") \
        .option("charset", "UTF-8") \
        .load(input_file_course)
    #course_df.printSchema()

    course_df = course_df.select("NAMHOC", "F_MAMH", "F_MAKH", "F_MANG", "F_TENMHVN", "F_DVHT", "TEN_KHKTVN", "KHOI", "COMPULSARY")
    # course_df.printSchema()
    
    course_df.createOrReplaceTempView("data")

    course_df = spark.sql(f"SELECT * FROM data WHERE (F_MAKH = '{faculty}' or F_MAKH = 'ALL')")
    # course_df.printSchema()
    
    course_df = course_df.withColumn("NAMHOC", course_df["NAMHOC"].cast(IntegerType())) \
        .withColumn("F_MAMH", course_df["F_MAMH"].cast(StringType())) \
        .withColumn("F_MAKH", course_df["F_MAKH"].cast(StringType())) \
        .withColumn("F_MANG", course_df["F_MANG"].cast(StringType())) \
        .withColumn("F_TENMHVN", course_df["F_TENMHVN"].cast(StringType())) \
        .withColumn("F_DVHT", course_df["F_DVHT"].cast(DecimalType())) \
        .withColumn("TEN_KHKTVN", course_df["TEN_KHKTVN"].cast(StringType())) \
        .withColumn("KHOI", course_df["KHOI"].cast(StringType())) \
        .withColumn("COMPULSARY", course_df["COMPULSARY"].cast(DecimalType())) 
    
    course_df.createOrReplaceTempView("data")
    
    course_stats_df = df.groupBy("F_MAMH").agg(
        F.mean("TKET").alias("avg_TKET"),            
        F.stddev("TKET").alias("std_TKET"),          
        (F.max("TKET") - F.min("TKET")).alias("range_TKET")     
    )
    
    pass_rate_df = df.groupBy("F_MAMH").agg(
        (F.sum(F.when(F.col("TKET") > 4, 1).otherwise(0)) / F.count("MASV") * 100).alias("PASS_RATE")
    )

    result_df = course_stats_df.join(pass_rate_df, on="F_MAMH", how="inner")

    result_df = result_df.withColumn(
        "LEVEL",
        F.when(
            (result_df["avg_TKET"] >= 6) 
            & (result_df["std_TKET"] <= 2) 
            & (result_df["PASS_RATE"] >= 70), 1)
        .when(
            (result_df["avg_TKET"] >= 5) 
            & (result_df["std_TKET"] <= 3)
            & (result_df["PASS_RATE"] >= 50), 2)  
        .otherwise(3) 
    )
    
    result_df = course_df.join(result_df, on="F_MAMH", how="left") \
                        .fillna(0, ["avg_TKET", "std_TKET", "PASS_RATE", "LEVEL"])

    # ulimit -n 188898 if too much open files errors
    result_df.coalesce(1).write\
        .mode("overwrite")\
        .option("header", "true")\
        .option("charset", "UTF-8")\
        .csv(output_path)
        
    return "--------- READ COURSE DONE ---------"
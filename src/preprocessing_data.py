from __future__ import print_function
from pyspark.sql.types import StringType, DoubleType, IntegerType, DecimalType
from pyspark.sql.functions import udf
import pyspark.sql.functions as spark_func
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def shouldZero(value):
    if value == "CT" or value == "VT" or value == "KD" or value == "RT":
        return True
    return False

def transform(x):
    print(x)
    a = isFloat(x)
    b = shouldZero(x)
    if b:
        return "0"
    else:
        if a:
            if float(x) > 10.0:
                return "0"
            else:
                return x
        else:
            return "REMOVE"

def preprocess(spark, input_file, output_path):
    # spark application
    spark.sparkContext.setLogLevel("WARN")
    
    # load original data
    df = spark.read.format("com.crealytics.spark.excel")\
        .option("useHeader", "true") \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("maxRowsInMemory", 2000) \
        .option("inferSchema", "false") \
        .option("addColorColumns", "False") \
        .option("charset", "UTF-8") \
        .load(input_file)
    #df.printSchema()
    
    # transform letter grade to number
    normalize_grade_udf = udf(lambda x: transform(x), StringType())

    df2 = df.select("NAMHOC", "NHHK", "F_MAMH", "F_TENMHVN", "F_DVHT", "F_MAKH", "F_MANG", "F_TENNGVN", "F_TENLOP", "MASV1", "TKET")\
        .withColumn("TKET", normalize_grade_udf(df["TKET"]))
    # df2.printSchema()
    
    df2.createOrReplaceTempView("data")

    df3 = spark.sql("SELECT * FROM data WHERE TKET <> 'REMOVE'")
    # df3.printSchema()
    
    df4 = df3.withColumn("TKET", df3["TKET"].cast(DoubleType())) \
        .withColumn("NAMHOC", df3["NAMHOC"].cast(IntegerType())) \
        .withColumn("F_DVHT", df3["F_DVHT"].cast(DecimalType())) \
        .withColumn("NHHK", df3["NHHK"].cast(IntegerType())) \
        .withColumn("F_MAKH", df3["F_MAKH"].cast(StringType())) \
        .withColumn("MASV", df3["MASV1"].cast(DecimalType())) 
        
    df4.createOrReplaceTempView("data")
    
    # df4.printSchema()

    window = Window.partitionBy([spark_func.col('F_MAMH'), spark_func.col('MASV')])\
        .orderBy(spark_func.col('TKET').desc())

    df5 = df4.select("*", spark_func.rank().over(window).alias("rank"))\
        .filter(spark_func.col('rank') <= 1)
    # df5.printSchema()
    
    # # ulimit -n 188898 if too much open files errors
    df5.coalesce(1).write\
        .mode("overwrite")\
        .option("header", "true")\
        .option("charset", "UTF-8")\
        .csv(output_path)
        
    return "--------- PRE-PROCESSING DATA DONE ---------"
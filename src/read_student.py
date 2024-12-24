from __future__ import print_function
import sys
from pyspark.sql.types import StringType,  IntegerType, DecimalType
from pyspark.sql.functions import udf
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

def read_student(input_path, output_path):
    # spark application
    spark = SparkSession.builder.appName("Read Student") \
        .config("spark.sql.debug.maxToStringFields", "20") \
        .getOrCreate()
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
        .load(input_path)
    #df.printSchema()
    
    # transform letter grade to number
    normalize_grade_udf = udf(lambda x: transform(x), StringType())

    df2 = df.select("NAMHOC", "F_MAMH", "F_MAKH", "F_MANG", "F_TENMHVN", "F_DVHT", "TEN_KHKTVN", "KHOI", "TKET")\
            .withColumn("TKET", normalize_grade_udf(df["TKET"]))
    # df2.printSchema()
    
    df2.createOrReplaceTempView("data")

    df3 = spark.sql("SELECT * FROM data WHERE TKET <> 'REMOVE'")
    # df3.printSchema()
    
    df4 = df3.withColumn("NAMHOC", df3["NAMHOC"].cast(IntegerType())) \
        .withColumn("F_MAMH", df3["F_MAMH"].cast(StringType())) \
        .withColumn("F_MAKH", df3["F_MAKH"].cast(StringType())) \
        .withColumn("F_MANG", df3["F_MANG"].cast(StringType())) \
        .withColumn("F_TENMHVN", df3["F_TENMHVN"].cast(StringType())) \
        .withColumn("F_DVHT", df3["F_DVHT"].cast(DecimalType())) \
        .withColumn("TEN_KHKTVN", df3["TEN_KHKTVN"].cast(StringType())) \
        .withColumn("KHOI", df3["KHOI"].cast(StringType())) \
        .withColumn("TKET", df3["TKET"].cast(DecimalType())) 
    df4.createOrReplaceTempView("data")
    # df4.printSchema()

    # ulimit -n 188898 if too much open files errors
    df4.coalesce(1).write\
        .mode("overwrite")\
        .option("header", "true")\
        .option("charset", "UTF-8")\
        .csv(output_path)
        
    print("--------- READ STUDENT DONE ---------")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: preprocessing_data.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    result = read_student(input_path, output_path)    
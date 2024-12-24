import glob
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType

if __name__ == "__main__":
    # loading input files - read-course, load all csv file - course
    path_course = "../data/read-course/*.csv"
    allcsv_course = glob.glob(path_course)
    input_file_course = allcsv_course
    # loading input files - read-course, load all csv file - student
    path_student = "../data/read-student/*.csv"
    allcsv_student = glob.glob(path_student)
    input_file_student = allcsv_student
    
    # var
    faculty = 'MT'
    speciality = 'KHM'
    subsector = 'KHKH'
    year = 2016
    
    # create spark session
    spark = SparkSession.builder.appName("Get Unlearned Courses").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    # Course Dataframe
    course_df = spark.read\
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file_course)
        
    course_df_2 = course_df.select("NAMHOC", "F_MAMH", "F_MAKH", "F_MANG", "F_DVHT", "COMPULSARY", "LEVEL")
    
    course_df_2.createOrReplaceTempView("data")

    course_df_3 = spark.sql(f"SELECT * FROM data WHERE (NAMHOC = '{year}') and (F_MAKH = '{faculty}' or F_MAKH = 'ALL') and (F_MANG = '{speciality}' or F_MANG = '{subsector}' or F_MANG = 'ALL')")
    
    course_df_4 = course_df_3.withColumn("F_MAMH", course_df_3["F_MAMH"].cast(StringType())) \
        .withColumn("F_MAKH", course_df_3["F_MAKH"].cast(StringType())) \
        .withColumn("F_MANG", course_df_3["F_MANG"].cast(StringType())) \
        .withColumn("F_DVHT", course_df_3["F_DVHT"].cast(IntegerType())) \
        .withColumn("COMPULSARY", course_df_3["COMPULSARY"].cast(IntegerType())) \
        .withColumn("LEVEL", course_df_3["LEVEL"].cast(IntegerType()))
    
    # course_df_4.show(20)
    
    # Student Dataframe
    student_df = spark.read\
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("charset", "UTF-8") \
        .csv(input_file_student)
        
    student_df_2 = student_df.select("F_MAMH", "F_MAKH", "F_MANG", "F_TENMHVN", "F_DVHT")
    
    student_df_2.createOrReplaceTempView("data")

    student_df_3 = spark.sql(f"SELECT * FROM data")
    
    student_df_4 = student_df_3.withColumn("F_MAMH", student_df_3["F_MAMH"].cast(StringType())) \
        .withColumn("F_MAKH", student_df_3["F_MAKH"].cast(StringType())) \
        .withColumn("F_MANG", student_df_3["F_MANG"].cast(StringType())) \
        .withColumn("F_TENMHVN", student_df_3["F_TENMHVN"].cast(StringType())) \
        .withColumn("F_DVHT", student_df_3["F_DVHT"].cast(IntegerType())) 
    student_df_4.createOrReplaceTempView("data")
    # student_df_4.show(20)
    
    unlearned_courses_df = course_df_4.join(student_df_4, 
                                        on=["F_MAMH", "F_MAKH", "F_MANG", "F_DVHT"], 
                                        how="left_anti")
    # unlearned_courses_df.show(20)
    
    unlearned_courses_df.coalesce(1).write\
        .mode("overwrite")\
        .option("header", "true")\
        .option("charset", "UTF-8")\
        .csv("../data/unlearned-courses")
        
    print("--------- UNLEARNED COURSE DONE ---------")
        
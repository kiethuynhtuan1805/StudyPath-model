import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.sql import SparkSession
from src import read_course as RC
# from src import read_student as RS
from src import preprocessing_data as PD
# from src import unlearned_courses as UC
from modules import main as p
from modules.neural_matrix_factorization import nmf as NMF
from modules.recommender_module.rcm import RCM as RCM


app = Flask(__name__)
CORS(app)

PD_FOLDER = "/app/data/pre-processed"
RC_FOLDER = "/app/data/read-course"
RS_FOLDER = "/app/data/read-student"
UC_FOLDER = "/app/data/unlearned-courses"
RAW_FOLDER = "/app/data/raw"

def upload():
    if 'file' not in request.files:
        return None, "No file part in the request"
    
    file = request.files['file']
    
    if file.filename == '':
        return None, "No file selected"
    
    try:
        file_path = os.path.join(RAW_FOLDER, file.filename)
        file.save(file_path)
        return file_path, None
    except Exception as e:
        return None, str(e) 

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    file_path, error = upload()
    if error:
        return jsonify({"status": "error", "message": error}), 400
    
    try:
        spark = SparkSession.builder.appName("Preprocessing data") \
            .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:3.5.1_0.20.4") \
            .config("spark.executor.memory", "8g") \
            .config("spark.total.executor.cores", "8") \
            .getOrCreate()
            # .master("spark://spark-master:7077") \
        result = PD.preprocess(spark, file_path, PD_FOLDER)
        spark.stop()
        return jsonify({"status": "success", "message": result}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/read-course', methods=['POST'])
def read_course():
    file_path, error = upload()
    faculties = request.form.getlist('faculty')
    if error:
        return jsonify({"status": "error", "message": error}), 400
    
    try:
        spark = SparkSession.builder.appName("Read Course") \
            .config("spark.sql.debug.maxToStringFields", "20") \
            .config("spark.executor.memory", "8g") \
            .config("spark.total.executor.cores", "8") \
            .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:3.5.0_0.20.3") \
            .getOrCreate()
            # .master("spark://spark-master:7077") \
            
        for faculty in faculties:
            OUTPUT_PATH = os.path.join(RC_FOLDER, faculty)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            result = RC.read_course(spark, PD_FOLDER, file_path, OUTPUT_PATH, faculty)
            
        spark.stop()
        return jsonify({"status": "success", "message": result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/preprocess-student', methods=['POST'])
def preprocess_student():
    return 

@app.route('/model-training', methods=['POST'])
def train():
    key = request.form.get('key')
    faculties = request.form.getlist('faculty')
    if key == "62f98ddd220aec5532aeb97dd55e2ef476ce81b8def3d563695510@@da596e53":
        pd_return = preprocess_data()
        spark = SparkSession.builder.appName("Neural Matrix Factorization")\
                    .config("spark.executor.memory", "8g") \
                    .config("spark.total.executor.cores", "8") \
                    .getOrCreate()
                    # .master("spark://spark-master:7077")\
                        
        for faculty in faculties:
            result = NMF.train_model(spark, PD_FOLDER, faculty)
            
        spark.stop()
        return jsonify({"status": "success", "message": result}), 200
    else:
        return jsonify({"message": "Access denied. Admins only."}), 403
    
@app.route('/model-predict', methods=['POST'])
def predict():
    data = request.json
    faculty = data.get('faculty')
    semester = data.get('semester')
    masv = data.get('masv')
    student_grade = data.get('student_grade')
    mang = data.get('mang')
    macn = data.get('macn')
    
    try:
        spark = SparkSession.builder.appName("Model Prediction") \
            .master("spark://spark-master:7077") \
            .config("spark.executor.memory", "8g") \
            .config("spark.total.executor.cores", "8") \
            .getOrCreate()
            
        result = p.main(spark, RC_FOLDER, faculty, mang, macn, semester, masv, student_grade, 128)
            
        spark.stop()
        return jsonify({"status": "TC", "message": result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

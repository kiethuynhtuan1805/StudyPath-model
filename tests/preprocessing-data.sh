PACKAGE=com.crealytics:spark-excel_2.12:3.5.0_0.20.3
URL=spark://26.177.92.120:7077
PROGRAM=../src/preprocessing-data.py
MEM=8G
CORES=8

# Command for running
spark-submit \
    --packages $PACKAGE \
    --master $URL \
    --executor-memory $MEM \
    --total-executor-cores $CORES \
    $PROGRAM
URL=spark://spark-master:7077
PROGRAM=/app/modules/neural_matrix_factorization/nmf.py
MEM=8G
CORES=8

# Command for running
spark-submit \
    --master $URL \
    --executor-memory $MEM \
    --total-executor-cores $CORES \
    $PROGRAM
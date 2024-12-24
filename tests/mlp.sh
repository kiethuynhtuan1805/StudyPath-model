URL=spark://26.177.92.120:7077
PROGRAM=../evaluation_modules/multi_layer_perceptron/mlp.py
MEM=8G
CORES=8
MODE=cluster

# Command for running
spark-submit \
    --master $URL \
    --executor-memory $MEM \
    --total-executor-cores $CORES \
    $PROGRAM
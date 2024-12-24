URL=spark://26.177.92.120:7077
PROGRAM=../modules/test.py
MEM=8G
CORES=8

# Command for running
spark-submit \
    --master $URL \
    --executor-memory $MEM \
    --total-executor-cores $CORES \
    $PROGRAM
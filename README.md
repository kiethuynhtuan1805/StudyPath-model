# HCMUT-Educational-Data-Analysis (HEDA) with Spark & Machine Learning

HEDA is a library/tool that can be used to analize educational data based on student's information and subject-scores that student have studied. Using HEDA, you can automatically gain:

- Data pre-processed
- The results of predicted scores for subjects that you have not studied yet
- ... to be continued

# Pre-requisite & How to run

1. Pre-requisite

If you have an account on the cluster - SuperNode-XP, you need to declare some ENV variables, as the following commands:

```bash
# On the testing envrionment of Spark on SuperNode-XP, Spark cluster is configured on 3 compute nodes
master  node3 (active)
slave1  node13 (active)

# Modify the .bashrc file
vim ~/.bashrc

# Add more lines to where Spark is installed
export SPARK_HOME=/opt/spark-3.4.3-bin-hadoop3
export PATH=$PATH:/opt/spark-3.4.3-bin-hadoop3/bin
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
export PATH=$SPARK_HOME/python:$PATH

# After everythign is done, we can access node3 (10.1.1.5) to submit jobs/tasks/apps to the Spark cluster

```

2. How to run & develop the program

This repository is solely maintained by HPC group.

```bash
# In the source code
/data   consists of
        /raw    : raw data files
        /pre-processed  : input files after pre-processing

/src    consists of preprocessing-data.py
/modules    consists of different sub-directories that are modules to analyze and predict something from pre-processed data.
/tests      consists of script files to run/test the source code
```

For exmple, to run/test the program, we can go into the folder - /tests

```bash
# The result of pre-processing-data.sh will be generated in /data/pre-processed
./preprocessing-data.sh

ls data/pre-processed   # the output like this
part-00000-5318a348-a32c-4aa6-a5eb-b1f4e95d3a71-c000.csv

```

3. Result

docker system prune -a --volumes

import subprocess
import sys
import time

import pymysql as mysql
import numpy as np

from datetime import datetime
from ..base.module import BaseANN

log_file_path = "create_index_time.log"

class OBVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = {"euclidean": "l2", "angular": "inner_product"}[metric]
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._conn = mysql.connect(host="127.0.0.1", user="root@perf", port=2881, passwd="", database="test")
        self._cur = self._conn.cursor()

        self._need_norm = False
        self._query = "SELECT id FROM items1 ORDER BY l2_distance(embedding, '[%s]') APPROXIMATE LIMIT %s"

    def normalize_vector(self, V):
        norm = np.linalg.norm(V)
        if norm == 0:
            norm = 1
        return V / norm

    def normalize_rows(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    def fit(self, X):
        np.random.seed(107)
        mean = 500
        std_dev = 100
        size = 1000000
        data = np.random.normal(mean, std_dev, size)
        data = np.random.normal(mean, std_dev, size)
        self._cur.execute("alter system set ob_vector_memory_limit_percentage=34;")
        self._cur.execute("DROP TABLE IF EXISTS items1")
        self._cur.execute(f"CREATE TABLE items1 (id int,c1 int, embedding vector({X.shape[1]}), primary key(id),key(c1))")
        self._cur.execute("set autocommit=1")
        print("copying data: data size: %d..." % X.shape[0])
        batch_size = 1000
        if self._need_norm:
            X = self.normalize_rows(X)
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            data_batch = [(i, X[i], data[i]) for i in range(start_idx, end_idx)]
            values = ["(%d,%d, '[%s]')" % (i,c1, ",".join([str(e) for e in embedding])) for i, embedding, c1 in data_batch]
            values_str = ",".join(values)
            self._cur.execute("insert into items1 values %s" % values_str)
        
        print("begin create index") 
        start_time = time.time()
        self._cur.execute(f"create vector index idx1 on items1(embedding) with (distance={self._metric}, type=hnsw, lib=vsag, m={self._m}, ef_construction={self._ef_construction})")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"create index time: {elapsed_time:.2f} seconds")
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Index creation time: {elapsed_time:.2f} seconds\n")
        print("begin major freeze") 
        self._cur.execute("ALTER SYSTEM MAJOR FREEZE;")
        time.sleep(2)
        count = 0
        while count != 1:
            self._cur.execute("SELECT COUNT(*) FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION WHERE STATUS = 'IDLE';")
            count = self._cur.fetchone()[0]
            print(f"Current count: {count}")
            if count != 1:
                time.sleep(5)
        print("major freeze end") 

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET ob_hnsw_ef_search = %d" % ef_search)

    def query(self, v, n):
        self._cur.execute(self._query % (','.join([str(ele) for ele in v]), n))
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        return 0

    def __str__(self):
        return f"OBVector(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

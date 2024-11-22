import argparse
import random
import pymysql
import time
import numpy as np

from ...runner import load_and_transform_dataset

def execute_and_fetch(cursor, query):
    cursor.execute(query)
    results = cursor.fetchall()
    return results

def calculate_recall(correct_result, result):
    if len(result) == 0:  # 避免除零错误
        return 0  # 当结果为空时，召回率定义为 0
    unmatched_count = sum(1 for item in result if item not in correct_result)
    recall = 1 - unmatched_count / len(result)
    return recall

def test_sql(cursor, ivf_sql, sql):
    correct_result = execute_and_fetch(cursor, ivf_sql)
 
    start_time = time.perf_counter()
    result = execute_and_fetch(cursor, sql)
    end_time = time.perf_counter()
    recall = calculate_recall(correct_result, result)
    execution_time = end_time - start_time
    return recall, execution_time

def run_queries(cursor, vector, limit, base_query, test_query, params):
    total_execution_time = 0
    for param in params:
        ivf_sql = base_query.format(vector, limit, param)
        sql = test_query.format(param, vector, limit)
        recall, execution_time = test_sql(cursor, ivf_sql, sql)
        total_execution_time += execution_time
    return recall, total_execution_time

def fit(cur, X):
    np.random.seed(107)
    mean = 500
    std_dev = 100
    size = 1000000
    data = np.random.normal(mean, std_dev, size)
    data = np.random.normal(mean, std_dev, size)
    cur.execute("alter system set ob_vector_memory_limit_percentage=34;")
    cur.execute("DROP TABLE IF EXISTS items1;")
    cur.execute(f"CREATE TABLE items1 (id int,c1 int, embedding vector({X.shape[1]}), primary key(id),key(c1));")
    cur.execute("set autocommit=1;")
    print("copying data: data size: %d..." % X.shape[0])
    batch_size = 10000
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X.shape[0])
        data_batch = [(i, X[i], data[i]) for i in range(start_idx, end_idx)]
        values = ["(%d,%d, '[%s]')" % (i,c1, ",".join([str(e) for e in embedding])) for i, embedding, c1 in data_batch]
        values_str = ",".join(values)
        cur.execute("insert into items1 values %s" % values_str)
    
    print("begin create index") 
    start_time = time.time()
    cur.execute(f"create vector index idx1 on items1(embedding) with (distance=l2, type=hnsw, lib=vsag, m=16, ef_construction=200);")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"create index time: {elapsed_time:.2f} seconds")
    print("begin major freeze") 
    cur.execute("ALTER SYSTEM MAJOR FREEZE;")
    time.sleep(2)
    count = 0
    while count != 1:
        cur.execute("SELECT COUNT(*) FROM oceanbase.DBA_OB_ZONE_MAJOR_COMPACTION WHERE STATUS = 'IDLE';")
        count = cur.fetchone()[0]
        print(f"Current count: {count}")
        if count != 1:
            time.sleep(5)
    print("major freeze end") 

def start_ann2(cur, X_test):
    try:
        items_processed = 0
        # 查询限制条件
        limit = 10000

        # 统计结果容器
        results_sql1 = []
        results_sql2 = []
        results_sql3 = []

        # 累计执行时间
        total_time_sql1 = 0
        total_time_sql2 = 0
        total_time_sql3 = 0

        for v in X_test[:10]:
            vector ="'[%s]'" % ','.join([str(ele) for ele in v])
            # Test SQL1
            base_query1 = "SELECT id FROM (SELECT id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {});"
            test_query1 = "SELECT * FROM (SELECT id FROM items1 ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {});"
            recall, exec_time = test_sql(cur, base_query1.format(vector, limit), test_query1.format(vector, limit))
            results_sql1.append((recall, exec_time))
            total_time_sql1 += exec_time
            

            # Test SQL2
            base_query2 = "SELECT id FROM (SELECT c1, id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {}) WHERE c1={};"
            test_query2 = "SELECT id FROM items1 WHERE c1={} ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {};"
            # 随机生成 `c1` 的值
            # 正态分布参数
            mean = 500          # 均值
            std_dev = 100       # 标准差
            size = 10           # 生成 4 个随机数

            # 使用正态分布生成随机数并限制在 [200, 801) 范围内
            params = [int(x) for x in np.clip(np.random.normal(mean, std_dev, size), 200, 800)]
            recall, exec_time = run_queries(cur, vector, limit, base_query2, test_query2, params)
            results_sql2.append((recall, exec_time))
            total_time_sql2 += exec_time

            # Test SQL3
            base_query3 = "SELECT * FROM (SELECT c1, id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {}) WHERE c1={};"
            test_query3 = "SELECT c1, id FROM items1 WHERE c1={} ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {};"
            recall, exec_time = run_queries(cur, vector, limit, base_query3, test_query3, params)
            results_sql3.append((recall, exec_time))
            total_time_sql3 += exec_time
            items_processed += 1
            if items_processed % 1 == 0:
                print("Processed %d/%d queries..." % (items_processed, len(X_test)))

        # Print results
        def print_results(query_name, results, total_time):
            print(f"\n{query_name} Results:")
            for idx, (recall, exec_time) in enumerate(results, start=1):
                exec_time = format(exec_time, '.6f')
                print(f"Run {idx}: Recall: {recall:.2f}, Query executed in: {exec_time} seconds")
            print(f"Total execution time for {query_name}: {total_time:.6f} seconds")

        print_results("SQL1", results_sql1, total_time_sql1)
        print_results("SQL2", results_sql2, total_time_sql2)
        print_results("SQL3", results_sql3, total_time_sql3)

    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_fit", 
        action="store_true", 
        help="If set, skip the fit process"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    connection = pymysql.connect(
        host="127.0.0.1",
        user="root@perf",
        port=2881,
        passwd="",
        database="test"
    )
    cursor = connection.cursor()
    X_train, X_test, distance = load_and_transform_dataset("sift-128-euclidean")
    if not args.skip_fit:
        fit(cursor, X_train)

    start_ann2(cursor, X_test)

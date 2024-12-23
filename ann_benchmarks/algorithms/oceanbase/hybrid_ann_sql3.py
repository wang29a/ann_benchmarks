import pymysql
import time

def execute_and_fetch(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def calculate_recall(correct_result, result):
    unmatched_count = sum(1 for item in result if item not in correct_result)
    recall = 1 - unmatched_count / len(result)
    return recall

def test_sql(cursor, ivf_sql, sql):
    # correct_result = execute_and_fetch(cursor, ivf_sql)
 
    start_time = time.perf_counter()
    result = execute_and_fetch(cursor, sql)
    end_time = time.perf_counter()
    # recall = calculate_recall(correct_result, result)
    recall = 0
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

def start_ann2():
    try:
        connection = pymysql.connect(
            host="127.0.0.1",
            user="root@perf",
            port=2881,
            passwd="",
            database="test"
        )
        cursor = connection.cursor()

        vector = str([65.0, 37.0, 2.0, 1.0, 16.0, 34.0, 14.0, 16.0, 15.0, 38.0, 45.0, 8.0, 26.0, 72.0, 21.0, 15.0, 35.0, 7.0, 13.0, 6.0, 4.0, 13.0, 11.0, 47.0, 8.0, 0.0, 0.0, 0.0, 26.0, 114.0, 7.0, 8.0, 108.0, 12.0, 1.0, 0.0, 5.0, 3.0, 14.0, 45.0, 81.0, 54.0, 71.0, 31.0, 7.0, 3.0, 7.0, 34.0, 8.0, 7.0, 52.0, 109.0, 33.0, 28.0, 5.0, 6.0, 3.0, 11.0, 4.0, 46.0, 119.0, 91.0, 2.0, 0.0, 109.0, 6.0, 6.0, 1.0, 0.0, 0.0, 2.0, 94.0, 86.0, 17.0, 4.0, 12.0, 5.0, 8.0, 41.0, 74.0, 8.0, 5.0, 7.0, 26.0, 40.0, 122.0, 56.0, 19.0, 24.0, 122.0, 18.0, 6.0, 37.0, 122.0, 7.0, 1.0, 54.0, 2.0, 20.0, 38.0, 8.0, 1.0, 3.0, 52.0, 74.0, 6.0, 8.0, 22.0, 25.0, 60.0, 35.0, 61.0, 36.0, 46.0, 2.0, 7.0, 70.0, 122.0, 9.0, 1.0, 62.0, 105.0, 8.0, 0.0, 2.0, 18.0, 5.0, 12.0])
        limit = 10000
        results = []

        # Test SQL1
        # base_query1 = "SELECT id FROM (SELECT id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {});"
        # test_query1 = "SELECT * FROM (SELECT id FROM items1 ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {});"
        # recall, exec_time = test_sql(cursor, base_query1.format(vector, limit), test_query1.format(vector, limit))
        # results.append(("SQL1", recall, exec_time))
        # # Test SQL2
        # base_query2 = "SELECT id FROM (SELECT c1, id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {}) WHERE c1={};"
        # test_query2 = "SELECT id FROM items1 WHERE c1={} ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {};"
        params = [270, 332, 387, 569]
        # recall, exec_time = run_queries(cursor, vector, limit, base_query2, test_query2, params)
        # results.append(("SQL2", recall, exec_time))

        # Test SQL3
        base_query3 = "SELECT * FROM (SELECT c1, id FROM items1 ORDER BY L2_distance(embedding, {}) LIMIT {}) WHERE c1={};"
        test_query3 = "SELECT c1, id FROM items1 WHERE c1={} ORDER BY L2_distance(embedding, {}) APPROXIMATE LIMIT {};"
        recall, exec_time = run_queries(cursor, vector, limit, base_query3, test_query3, params)
        results.append(("SQL3", recall, exec_time))

        # Print results
        for query_name, recall, exec_time in results:
            exec_time =  format(exec_time, '.6f')
            print(f"{query_name} -> Recall: {recall}, Query executed in: {exec_time} seconds")

    except pymysql.MySQLError as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    start_ann2()

from clickhouse_driver import Client


class ClickhouseParserCls():
    def __init__(self, ip, port, server_name, user, pswd):
        self.ip = ip
        self.port = port
        self.server = server_name
        self.user = user
        self.pswd = pswd

    def connect(self):
        try:
            connect = Client(host=self.ip, port=self.port, database=self.server, user=self.user, password=self.pswd)
            return connect
        except Exception as e:
            print("connect to clickhouse database cause error!!\n")
            print("the detail is %s" % str(e))
            return str(e)

    def executeSql(self, sql):
        try:
            conn = self.connect()
            result = conn.execute(sql)
            conn.disconnect()
            return result
        except Exception as e:
            print('fail to execute sql, the detail is {}'.format(str(e)))
            return str(e)


if __name__ == '__main__':
    db_name = 'test'
    user_name = 'default'
    password = 'default'
    host = '10.10.201.15'
    port = '9000'
    pg_cls = ClickhouseParserCls(host, port, db_name, user_name, password)
    sql = 'select * from t_tpi_intersection_delay2'
    data = pg_cls.executeSql(sql)
    print(data)

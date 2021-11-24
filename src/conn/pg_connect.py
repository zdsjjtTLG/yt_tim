from sqlalchemy import create_engine
import pandas as pd

class OperatePgSql:
    def __init__(self, db_type, drivertype, username, password, host, port, databasename):
        self.db_type = db_type
        self.drivertype = drivertype
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.databasename = databasename

    def get_con(self):
        try:
            connect_string = self.db_type + '+' + self.drivertype + '://' + self.username + ':' + self.password + '@' + self.host + ':' + self.port + '/' + self.databasename
            engine = create_engine(connect_string)
            return engine
        except Exception as e:
            print("Connect PgSql Error:", e)


if __name__ == '__main__':
    db_type = 'postgresql'
    drivertype = 'psycopg2'
    username = 'postgres'
    password = '123456'
    host = '10.10.5.108'
    port = '5432'
    databasename = 'tpi_yt'

    pgsql = OperatePgSql(db_type, drivertype, username, password, host, port, databasename)
    engine = pgsql.get_con()
    trip_data = pd.read_sql('select * from t_base_link_geom', engine)
    print(trip_data)

import sqlite3
import pickle

DEFAULT_LOG_COLUMNS = [
    'Youbot1',
    'Youbot2',
    'Youbot1_d',
    'Youbot2_d',
    'ee1', 'ee2',
    'nl12', 'de1',
    'nl21', 'de2',
    'tube'
]

class Log:
    def __init__(self, name:str, columns:list[str]=DEFAULT_LOG_COLUMNS, replace:bool=False) -> None:
        self.name = name
        self.conn = sqlite3.connect(f'logging/logs.db')
        self.cursor = self.conn.cursor()
        columns_definition = ', '.join([f"{col} BLOB" for col in columns])
        # Create a table if it doesn't exist
        if replace:
            self.cursor.execute(f'''DROP TABLE IF EXISTS {name}''')
        self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {name} (time REAL, {columns_definition})''')

    def write(self, colNames:list[str], values:list) -> None:
        blobs = [pickle.dumps(dat) for dat in values[1:]]
        cols = ', '.join([c for c in colNames])
        vals = ', '.join(['?' for c in colNames])
        self.cursor.execute(f'''
            INSERT INTO {self.name} ({cols}) VALUES ({vals})
        ''', (values[0], *blobs))
    
    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


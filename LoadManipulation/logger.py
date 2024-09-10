import sqlite3
import pickle


columnsDD = [[f'q{i}', f'dq{i}', f'ee{i}', f'rho{i}', f'zeta{i}', f'EQ{i}', f'IQ{i}', f'SC_eex{i}', f'SC_eez{i}'] for i in range(4)]
DEFAULT_LOG_COLUMNS  = [item for sublist in columnsDD for item in sublist]
DEFAULT_LOG_COLUMNS += ['loadP', 'loadR', 'SC_loadx', 'SC_loady']
DEFAULT_LOG_COLUMNS += ['desiredP', 'desiredR']
DEFAULT_LOG_COLUMNS += ['nl01', 'nl12', 'nl23', 'nl03', 'K01', 'K12', 'K23', 'K03']

class Log:
    def __init__(self, name:str, columns:list[str]=DEFAULT_LOG_COLUMNS, replace:bool=False) -> None:
        self.name = name
        self.conn = sqlite3.connect(f'logging/logs.db')
        self.cursor = self.conn.cursor()
        columns_definition = ', '.join([f"{col} BLOB" for col in columns])
        # Create a table if it doesn't exist
        if replace:
            self.cursor.execute(f'''DROP TABLE IF EXISTS {name}''')
        self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {name} (time REAL PRIMARY KEY, {columns_definition})''')

    def write(self, colNames:list[str], values:list) -> None:
        blobs = [pickle.dumps(dat) for dat in values[1:]]
        cols = ', '.join([c for c in colNames])
        vals = ', '.join(['?' for c in colNames])
        update_columns = ', '.join(f"{c} = excluded.{c}" for c in colNames)
        self.cursor.execute(f'''
            INSERT INTO {self.name} ({cols}) VALUES ({vals})
            ON CONFLICT (time) DO UPDATE SET {update_columns}
        ''', (values[0], *blobs))
    
    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


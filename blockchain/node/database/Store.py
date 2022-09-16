import sqlite3


class Database:
    _con = None
    _curObj = None

    def __init__(self):
        self._con = sqlite3.connect('blockDatabase.db')
        self._curObj = self._con.cursor()
        self._curObj.execute('CREATE table')

    def insert(self):
        pass

    def delete(self):
        pass

    def query(self):
        pass


if __name__ == '__main__':
    Database()

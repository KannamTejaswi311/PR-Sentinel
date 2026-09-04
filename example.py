import sqlite3

def get_user(username):
    connection = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    result = connection.execute(query)
    return result.fetchone()

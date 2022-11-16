import mysql.connector



def insert_platform(sql_cursor,platform_id,platform_name):
    '''
    Insert a unique platform (id,name) into platform_type table
    
    parameter sql_cursor: mysql_connector.cursor object
    parameter platform_id: id for the platform
    parameter platform_name: name of the platform (e.g., Tumblr)
    '''
    cmd_str = ("INSERT INTO platfrom_type"
               "(platform_type_name) "
               "VALUES (%s)")
    res = cursor.execute(cmd_str, (platform_name))
    cnx.commit()
    cnx.close()

    return res
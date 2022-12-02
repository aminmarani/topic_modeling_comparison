import mysql.connector



def insert_platform(sql_connector,platform_id,platform_name):
    '''
    Insert a unique platform (id,name) into platform_type table
    
    parameter sql_connector: mysql_connector.connect object
    parameter platform_id: id for the platform
    parameter platform_name: name of the platform (e.g., Tumblr)
    '''
    cmd_str = ("INSERT INTO platform_type"
               "(platform_type_name) "
               "VALUES (%s)")
    
    cursor = sql_connector.cursor()
    res = cursor.execute(cmd_str, (platform_name,))
    sql_connector.commit()
    sql_connector.close()

    return res
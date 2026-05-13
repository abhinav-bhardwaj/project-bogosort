from app.db.article_repository import initialize_schema, setup_database

def init_db(sql_uri=None):
    if sql_uri:
        setup_database(sql_uri)
    initialize_schema()
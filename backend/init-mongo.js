db.auth('kdb_user', '1234')

db = db.getSiblingDB('kdb')

db.createUser({
    user: 'kdb_user',
    pwd: '1234',
    roles: [
        {
            role: 'readWrite',
            db: 'kdb'
        }
    ]
}) 
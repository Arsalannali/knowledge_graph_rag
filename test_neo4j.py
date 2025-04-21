from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError

def test_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    
    print(f"Attempting to connect to Neo4j at {uri} with user {user}")
    
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            print(f"Connection successful! Test query result: {record['num']}")
            
            # Try to get node count
            result = session.run("MATCH (n) RETURN count(n) AS count")
            record = result.single()
            print(f"Total nodes in database: {record['count']}")
        
        driver.close()
        print("Connection closed.")
        return True
    except AuthError as e:
        print(f"Authentication failed: {e}")
        return False
    except ServiceUnavailable as e:
        print(f"Neo4j service unavailable: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    test_connection() 
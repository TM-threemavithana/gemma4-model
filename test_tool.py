import asyncio
import logging

# Ensure logging is visible
logging.basicConfig(level=logging.INFO)

async def run_test():
    from src.tools.registry import query_project_echo
    
    print("--------------------------------------------------")
    print("Testing query_project_echo tool directly...")
    print("--------------------------------------------------")
    
    # Let's ask a question an outside caller would ask
    test_query = "Hi, can I get the details on your cheapest package available?" 
    
    print(f"Sending query: '{test_query}'")
    
    # Run the tool!
    result = await query_project_echo(test_query)
    
    print("--------------------------------------------------")
    print("Result from Project Echo:")
    print(result)
    print("--------------------------------------------------")

if __name__ == "__main__":
    asyncio.run(run_test())

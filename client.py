import asyncio
import nest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import base64

nest_asyncio.apply()  # Needed to run interactive Python

async def main():
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # Make sure this points to your server file
    )

    # Connect to the server
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("Client connected to server.")

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for idx, tool in enumerate(tools_result.tools, 1):
                print(f"{idx}. {tool.name}: {tool.description}")

            # Call calculator tool
            result_add = await session.call_tool("add", arguments={"a": 5, "b": 7})
            print("Add result (5 + 7):", result_add.content[0].text)
            print("\n")


            # Call Wikipedia search tool
            result_wikipedia = await session.call_tool("wikipedia_search", arguments={"query": "India's GDP"})
            print("Wikipedia search result:")
            print(result_wikipedia.content[0].text)   
            print("\n")   

            # Call Wikidata search tool
            result_wikidata = await session.call_tool("wikidata_search", arguments={"query": "India's GDP"})
            print("Wikidata search result:")
            print(result_wikidata.content[0].text)   
            print("\n")

            # Call Semantic scholar search tool
            result_semantic_scholar = await session.call_tool("semantic_scholar_search", arguments={"query": "India's GDP"})
            print("Semantic scholar search result:")
            print(result_semantic_scholar.content[0].text)   
            print("\n")

            # Call PubMed search tool
            result_pubMed = await session.call_tool("pubmed_search", arguments={"query": "COVID-19 vaccine efficacy"})
            print("PubMed search result:")
            print(result_pubMed.content[0].text)   
            print("\n")
            
if __name__ == "__main__":
    asyncio.run(main())

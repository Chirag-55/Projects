import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pandas as pd
import gradio as gr
import requests
import asyncio
import base64
import tempfile
import pytz
import aiohttp
from tempfile import TemporaryDirectory 
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
from dotenv import load_dotenv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.tools.searx_search.tool import SearxSearchRun
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun  
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.tools import YouTubeSearchTool
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen.coding import LocalCommandLineCodeExecutor
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.openbb import OpenBBTools
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_experimental.utilities import PythonREPL
from langchain_exa import ExaSearchRetriever
from smolagents.default_tools import PythonInterpreterTool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from smolagents.default_tools import SpeechToTextTool
from smolagents.default_tools import VisitWebpageTool
from smolagents.default_tools import UserInputTool
from tavily import TavilyClient
from readability.readability import Document
from bs4 import BeautifulSoup
from markdownify import markdownify as md



load_dotenv("../.env")

# Create the MCP server
mcp = FastMCP(
    name="CalculatorAndSearch",
    host="0.0.0.0",
    port=8050,
)

# Add a simple calculator tool
@mcp.tool(
    name="add",
    description="Add two numbers together."
)
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Initialize DuckDuckGo tool
ddg_tool = DuckDuckGoSearchRun()

# Add DuckDuckGo web search tool
@mcp.tool(
    name="duckduckgo_search",
    description="Search the web using DuckDuckGo and return the top result."
)
def duckduckgo_search(query: str) -> str:
    """Perform a web search using DuckDuckGo."""
    return ddg_tool.run(query)

# Initialize Searx tool wrapper and run tool
searx_wrapper = SearxSearchWrapper(searx_host="https://searx.info")
searx_tool = SearxSearchRun(wrapper=searx_wrapper)

@mcp.tool(
    name="searx_search",
    description="Search the web using a SearxNG instance and return the top result."
)
def searx_search(query: str) -> str:
    """Perform a web search using Searx."""
    return searx_tool.run(query)

# Initialize Google Serper search
serper_api = GoogleSerperAPIWrapper(serper_api_key="xxx")
serper_tool = GoogleSerperRun(api_wrapper=serper_api)

@mcp.tool(
        name="google_serper_search", 
        description="Search using Google via Serper API.")
def google_serper_search(query: str) -> str:
    return serper_tool.run(query)

# Initialize Arxiv search tool
arxiv_tool = ArxivQueryRun()

@mcp.tool(
    name="arxiv_search",
    description="Search for academic papers on Arxiv and return relevant results."
)
def arxiv_search(query: str) -> str:
    """Perform a search for academic papers on Arxiv."""
    return arxiv_tool.run(query)

# Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, lang="en")  # Optional customization
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

@mcp.tool(
    name="wikipedia_search",
    description="Search Wikipedia for a topic and return a summary."
)
def wikipedia_search(query: str) -> str:
    return wikipedia_tool.run(query)


# Initialize Wikidata API wrapper
wikidata_wrapper = WikidataAPIWrapper()
wikidata_tool = WikidataQueryRun(api_wrapper=wikidata_wrapper)

@mcp.tool(
    name="wikidata_search",
    description="Search Wikidata for entities and facts and return structured information."
)
def wikidata_search(query: str) -> str:
    return wikidata_tool.run(query)

# Semantic Scholar Search Tool
semantic_scholar_tool = SemanticScholarQueryRun(api_wrapper=SemanticScholarAPIWrapper())

@mcp.tool(
        name="semantic_scholar_search", 
        description="Search academic papers via Semantic Scholar."
        )
def semantic_scholar_search(query: str) -> str:
    return semantic_scholar_tool.run(query)

# Initialize PubMed search tool using PubmedQueryRun
pubmed_tool = PubmedQueryRun()

@mcp.tool(
    name="pubmed_search",
    description="Search for academic papers on PubMed and return relevant results."
)
def pubmed_search(query: str) -> str:
    """Perform a search for academic papers on PubMed."""
    return pubmed_tool.run(query)

# Create a temporary directory for file operations
working_directory = TemporaryDirectory()

# Function to handle file operations manually
def write_file(file_name: str, content: str) -> str:
    """Write content to a file in the temporary directory."""
    file_path = os.path.join(working_directory.name, file_name)
    with open(file_path, 'w') as f:
        f.write(content)
    return f"File {file_name} created with content: {content}"

def read_file(file_name: str) -> str:
    """Read content from a file in the temporary directory."""
    file_path = os.path.join(working_directory.name, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    return f"File {file_name} not found."

def delete_file(file_name: str) -> str:
    """Delete a file in the temporary directory."""
    file_path = os.path.join(working_directory.name, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return f"File {file_name} deleted."
    return f"File {file_name} not found."

# Define the filesystem tool
@mcp.tool(
    name="filesystem_tool",
    description="Use this tool to manage files in a temporary directory."
)
def filesystem_tool(action: str, file_name: str = None, content: str = None) -> str:
    """Interact with the file system, such as creating or reading files."""
    if action == "create" and file_name and content:
        return write_file(file_name, content)
    elif action == "read" and file_name:
        return read_file(file_name)
    elif action == "delete" and file_name:
        return delete_file(file_name)
    else:
        return "Invalid action or missing parameters."


#  Youtube Search Tool

youtube_tool = YouTubeSearchTool()

@mcp.tool(
    name="youtube_search",
    description="Search YouTube for relevant videos and return the top result."
)
def youtube_search(query: str) -> str:
    return youtube_tool.run(query)

# Python Code Executor Tool
executor = LocalCommandLineCodeExecutor()
python_exec_tool = PythonCodeExecutionTool(executor=executor)


@mcp.tool(
    name="python_code_execution",
    description="Execute Python code on the local command line and return the result."
)
async def python_code_execution(code: str) -> str:
    """Execute arbitrary Python code via local command-line execution."""
    try:
        # Execute the Python code
        result = await python_exec_tool.run(code, cancellation_token=None)  # Await the execution
        print(f"Execution result: {result}")  # Log for debugging
        return result if result else "No output or an error occurred."
    except Exception as e:
        print(f"Error executing code: {e}")
        return f"Error: {e}"


# Firecrawl Tool

firecrawl_tool = FirecrawlTools(api_key="fc-xx")

@mcp.tool(
    name="firecrawl_scrape",
    description="Scrape the given webpage using Firecrawl and return its body content."
)
async def firecrawl_scrape(url: str) -> str:
    """Scrape webpage content using Firecrawl."""
    try:
        # Assuming scrape_website is synchronous, so no await is needed
        result = firecrawl_tool.scrape_website(url)
        
        return result if result else "No content returned."
    except Exception as e:
        return f"Error: {e}"


#  OpenBB tools 
openbb_tool = OpenBBTools(provider="yfinance", company_news=True)


@mcp.tool(
    name="openbb_macro_data",
    description="Retrieve macroeconomic data using OpenBB, such as GDP or inflation."
)
def openbb_macro_data(query: str) -> str:
    """Fetch macroeconomic data using OpenBB Tools."""
    try:
        return openbb_tool.get_company_news(query)
        return result if result else "No data returned."
    except Exception as e:
        return f"Error: {e}"
    
# scrape_webpage Tool
@mcp.tool(
    name="scrape_webpage",
    description="Scrape the webpage using async HTTP client and return its HTML content."
)
async def scrape_webpage(url: str) -> str:
    """Scrape the given URL asynchronously using aiohttp."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                html = await resp.text()
                return html if html else "No content returned."
    except Exception as e:
        return f"Error scraping the webpage: {e}"
    
    
# EXA SEARCH TOOL
exa_search_retriever = ExaSearchRetriever(api_key="XXX")

@mcp.tool(
    name="exa_search",
    description="Search Exa for relevant documents based on a query."
)
def exa_search(query: str) -> str:
    # Use the correct method
    docs = exa_search_retriever.get_relevant_documents(query)
    
    # Combine the text content into a single string
    return "\n\n".join(doc.page_content for doc in docs)

#  PYTHON INTERPRETER TOOL
python_interpreter_tool = PythonInterpreterTool()

@mcp.tool(
    name="python_interpreter",
    description="Executes Python code using the SmolAgents' Python interpreter."
)
def python_interpreter(code: str) -> str:
    """Executes Python code using the SmolAgents' Python interpreter tool."""
    if not isinstance(code, str):
        raise ValueError("Input code must be a string.")
    
    try:
        result = python_interpreter_tool.forward(code)
        return result
    except Exception as e:
        return f"Execution error: {e}"
    
    #Final Answer Tool
@mcp.tool(
    name="final_answer",
    description="Provides a final answer to the given problem."
)
def final_answer(answer: str) -> str:
    return answer

    #GraphQL Tool

GRAPHQL_ENDPOINT = "https://countries.trevorblades.com/"

@mcp.tool(
    name="graphql_query",
    description="Executes a GraphQL query and returns the result as JSON."
)
def graphql_query(query: str) -> str:
    """
    Execute a GraphQL query and return the result as a JSON-formatted string.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            GRAPHQL_ENDPOINT,
            json={"query": query},
            headers=headers
        )
        response.raise_for_status()
        return response.text  # or `json.dumps(response.json(), indent=2)` for pretty output
    except requests.exceptions.RequestException as e:
        return f"GraphQL query failed: {e}"
    
#  User Input Tool
@mcp.tool(
    name="user_input",
    description="Prompts the user for input and returns their response."
)
def user_input(prompt: str) -> str:
    tool = UserInputTool()
    return tool(prompt)

# youtube_transcript_scraper TOOL
@mcp.tool(
    name="youtube_transcript_scraper",
    description="Extracts transcript text from a YouTube video given its URL."
)
def youtube_transcript_scraper(video_url: str) -> str:
    """Scrape transcript from a YouTube video URL."""
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry["text"] for entry in transcript])
        return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound):
        return "No transcript available for this video."
    except Exception as e:
        return f"Failed to fetch transcript: {str(e)}"
    
# Initialize the wrapper and toolkit
text_requests_wrapper = TextRequestsWrapper()
requests_toolkit = RequestsToolkit(requests_wrapper=text_requests_wrapper)

# Wrap as an MCP tool
@mcp.tool(
    name="http_get_request",
    description="Make an HTTP GET request to a given URL and return the text response."
)
def http_get_request(url: str) -> str:
    """Performs an HTTP GET request using LangChain's RequestsToolkit."""
    tool = requests_toolkit.get_tools()[0]  # Typically the GET tool is first
    return tool.invoke({"url": url})

# Define the JSON Schema as a dictionary
json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "key": {
            "type": "string"
        },
        "value": {
            "type": "string"
        }
    },
    "required": ["key", "value"]
}

# Initialize JsonSpec by passing the schema as the dict_ field
json_spec = JsonSpec(dict_=json_schema)

# Initialize JsonToolkit with the schema
json_toolkit = JsonToolkit(spec=json_spec)

@mcp.tool(
    name="parse_json_response",
    description="Parse a JSON string and return the parsed result."
)
def parse_json_response(json_string: str) -> dict:
    """Parses the provided JSON string using the LangChain JSON Toolkit."""
    return json_toolkit.invoke({"json_string": json_string})


    
# Date and Time tool (similar to AgentIQ)
now = datetime.now(pytz.timezone("Asia/Kolkata"))

@mcp.tool(
    name="get_datetime",
    description="Get the current date and time in a readable format like 'Tuesday, May 27, 2025 at 02:32 PM'."
)
def get_datetime() -> str:
    """Return the current date and time in a human-readable format."""
    now = datetime.now()  # or use pytz.timezone('Asia/Kolkata') if you want specific timezone
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

#  SmolAgents transcription tool 

transcriber_tool = SpeechToTextTool()

@mcp.tool(
    name="transcriber",
    description="Transcribe an audio file using SmolAgents’ Whisper-based tool."
)
def transcriber(audio_path: str) -> str:
    """
    Transcribe audio using SmolAgents' SpeechToTextTool.

    Args:
        audio_path (str): Path to the audio file (local).
    
    Returns:
        str: Transcribed text.
    """
    return transcriber_tool(audio=audio_path)

#Tavily Search Tool

TAVILY_API_KEY = "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"
client = TavilyClient(api_key=TAVILY_API_KEY)

@mcp.tool(
    name="tavily_search",
    description="Search the web using Tavily API and return top result snippet and URL."
)
def tavily_search(query: str) -> str:
    """
    Perform a search using Tavily and return the first result.
    """
    results = client.search(query=query, search_depth="basic", include_answer=True)
    if results and results.get("results"):
        top = results["results"][0]
        return f"{top['title']}\n{top['url']}\n\n{top['content']}"
    return "No results found."

# Webpage_scraper Tool

@mcp.tool(
    name="webpage_scraper",
    description="Scrape a webpage and return the readable content in markdown format."
)
def webpage_scraper(url: str) -> str:
    """Fetch and clean a webpage, returning content as markdown."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    doc = Document(response.text)
    readable_html = doc.summary()
    soup = BeautifulSoup(readable_html, "html.parser")
    markdown = md(str(soup))
    return markdown

#  Visit WebPage tool
@mcp.tool(
    name="visit_webpage",
    description="Fetches and returns the plain text content of a webpage."
)
def visit_webpage(url: str) -> str:
    tool = VisitWebpageTool()
    return tool(url)


# Run the server
if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")

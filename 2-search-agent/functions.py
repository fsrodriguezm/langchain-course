from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient()

@tool
def tavily_search(
    query: str,
    time_range: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Query Tavily for current info; supports either time_range or start/end dates."""

    if time_range and (start_date or end_date):
        start_date = None
        end_date = None

    valid_time_ranges = {"day", "week", "month", "year", "d", "w", "m", "y"}
    if time_range and time_range.lower() not in valid_time_ranges:
        time_range = None

    params = {
        "query": query,
        "time_range": time_range,
        "start_date": start_date,
        "end_date": end_date,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 5,
    }
    clean_params = {k: v for k, v in params.items() if v not in (None, [], "")}
    return tavily_client.search(**clean_params)

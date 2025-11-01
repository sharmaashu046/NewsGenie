import streamlit as st
import asyncio
from typing import TypedDict, Union, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.tools import tool
import requests
import json
import nest_asyncio

# Apply asyncio patch for Streamlit compatibility
nest_asyncio.apply()

# --- Configuration ---
# IMPORTANT: Replace these with your actual API keys
GROQ_API_KEY = "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"
GNEWS_API_KEY = "ea5356372a932f33d835cde847e7c6a2"
SERPER_API_KEY = "ef93b9a07a6d79d235d1dde91049d06c04bc79d3"


# Validate API keys
API_KEYS_LOADED = all([
    GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_KEY",
    GNEWS_API_KEY and GNEWS_API_KEY != "YOUR_GNEWS_KEY",
    SERPER_API_KEY and SERPER_API_KEY != "YOUR_SERPER_KEY"
])

# --- Tool Definitions ---
@tool
def get_news(query: str) -> str:
    """Searches for news articles matching the query."""
    if not API_KEYS_LOADED:
        return "Error: GNews API Key not configured."
    
    from urllib.parse import quote
    
    # URL-encode the query properly
    encoded_query = quote(query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&max=5&apikey={GNEWS_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        
        if not articles:
            return f"No recent news found for: {query}\n\nTry a different search term or broader topic."
        
        results = []
        for i, a in enumerate(articles):
            title = a.get('title', 'N/A')
            source = a.get('source', {}).get('name', 'N/A')
            description = a.get('description', '')
            # Truncate description if too long
            if description and len(description) > 150:
                description = description[:150] + "..."
            
            result_text = f"{i+1}. **{title}**\n   📰 {source}"
            if description:
                result_text += f"\n   {description}"
            results.append(result_text)
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error fetching news: {e}"

@tool
def web_search(query: str) -> str:
    """Performs a web search for the given query and returns top results."""
    if not API_KEYS_LOADED:
        return "Error: Serper API Key not configured."
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'content-type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json().get('organic', [])
        
        if not results:
            return f"No web search results found for: {query}"
        
        snippets = [f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}" 
                   for res in results[:3]]
        return "\n---\n".join(snippets)
    except Exception as e:
        return f"Error performing web search: {e}"

# --- LLM Initialization ---
llm = None
if API_KEYS_LOADED:
    try:
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",  # Using a more reliable model
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        st.sidebar.error(f"Groq Init Error: {e}")

# --- LangGraph State Definition ---
class NewsGenieState(TypedDict):
    user_query: str
    news_category: str
    query_type: str
    news_results: str
    search_results: str
    final_response: str
    error_message: Union[str, None]

# --- Node Definitions ---
class QueryClassifier(BaseModel):
    query_type: Literal["news_request", "general_query"]

async def classify_query_node(state: NewsGenieState):
    """Classify user query as news request or general query."""
    if not llm:
        return {"query_type": "general_query", "error_message": "LLM unavailable."}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the query as 'news_request' or 'general_query'. "
                  "News categories: technology, finance, sports. "
                  "Respond ONLY with the QueryClassifier JSON."),
        ("human", "Query: '{query}'.")
    ])
    
    classifier_chain = prompt | llm.with_structured_output(QueryClassifier)
    
    try:
        result = await classifier_chain.ainvoke({"query": state['user_query']})
        return {"query_type": result.query_type, "error_message": None}
    except Exception as e:
        return {"query_type": "general_query", "error_message": f"Classification failed: {e}"}

async def fetch_news_node(state: NewsGenieState):
    """Fetch news headlines based on user query."""
    user_query = state.get('user_query', '')
    
    # Extract search terms from the query
    # Simple keyword extraction - you can make this smarter
    search_query = user_query.lower()
    
    # Remove common question words to get core topic
    remove_words = ['show me', 'get me', 'find', 'search for', 'news on', 'news about', 
                    'headlines on', 'headlines about', 'latest', 'recent', 'suggest']
    for word in remove_words:
        search_query = search_query.replace(word, '')
    
    search_query = search_query.strip()
    
    # If query is too short or empty, use category fallback
    if len(search_query) < 3:
        category = state.get('news_category', 'technology')
        search_query = category
    
    try:
        news_result = await asyncio.to_thread(get_news.invoke, {"query": search_query})
        return {"news_results": news_result, "error_message": None}
    except Exception as e:
        return {"error_message": f"News fetch failed: {e}"}

async def general_query_node(state: NewsGenieState):
    """Handle general queries with web search support."""
    if not llm:
        return {"final_response": "LLM unavailable.", "error_message": "LLM unavailable."}
    
    user_query = state['user_query']
    search_results_text = "(Web search not performed)"
    
    # Perform web search
    try:
        search_results_val = await asyncio.to_thread(web_search.invoke, {"query": user_query})
        if "Error:" not in search_results_val:
            search_results_text = f"\n\nRelevant Web Search Info:\n{search_results_val}"
    except Exception as e:
        print(f"Web Search Error: {e}")
    
    # Generate answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are NewsGenie, a helpful AI assistant. "
                  "Answer the user's query concisely using web search info if available."),
        ("human", "Query: '{query}'{search_info}")
    ])
    
    answer_chain = prompt | llm | StrOutputParser()
    
    try:
        answer = await answer_chain.ainvoke({
            "query": user_query,
            "search_info": search_results_text
        })
        return {"final_response": answer, "error_message": None}
    except Exception as e:
        return {
            "error_message": f"Answer generation failed: {e}",
            "final_response": "Sorry, I couldn't generate an answer."
        }

async def format_response_node(state: NewsGenieState):
    """Format the final response based on query type and results."""
    error = state.get('error_message')
    query_type = state.get('query_type')
    
    if error:
        final_answer = f"⚠️ An error occurred: {error}"
    elif query_type == "news_request":
        news = state.get('news_results', 'Could not retrieve news.')
        if "Error:" in news or "No recent news found" in news:
            final_answer = f"📰 {news}"
        else:
            final_answer = f"📰 **News Results:**\n\n{news}"
    elif state.get("final_response"):
        final_answer = state["final_response"]
    else:
        final_answer = "Sorry, I couldn't process your request."
    
    return {"final_response": final_answer}

# --- Routing Logic ---
def route_after_classification(state: NewsGenieState):
    """Route to appropriate node based on classification."""
    if state.get("error_message"):
        return "format_response"
    return "fetch_news" if state.get("query_type") == "news_request" else "general_query"

# --- Build LangGraph Workflow ---
workflow = StateGraph(NewsGenieState)

# Add nodes
workflow.add_node("classify_query", classify_query_node)
workflow.add_node("fetch_news", fetch_news_node)
workflow.add_node("general_query", general_query_node)
workflow.add_node("format_response", format_response_node)

# Set entry point
workflow.set_entry_point("classify_query")

# Add edges
workflow.add_conditional_edges(
    "classify_query",
    route_after_classification,
    {
        "fetch_news": "fetch_news",
        "general_query": "general_query",
        "format_response": "format_response"
    }
)
workflow.add_edge("fetch_news", "format_response")
workflow.add_edge("general_query", "format_response")
workflow.add_edge("format_response", END)

# Compile graph
try:
    app_graph = workflow.compile()
except Exception as e:
    st.error(f"Graph compilation error: {e}")
    app_graph = None

# --- Streamlit UI ---
st.set_page_config(page_title="NewsGenie AI", page_icon="📰", layout="wide")

st.title("📰 NewsGenie AI Assistant")
st.caption("Your AI for news headlines and general questions")

# Sidebar
st.sidebar.header("📰 News Categories")
news_category = st.sidebar.radio(
    "Select a category for news headlines:",
    ("Technology", "Finance", "Sports"),
    index=0
)

st.sidebar.info("💡 **Tips:**\n- Try: 'Premier League news', 'AI technology updates'\n- Ask general questions for AI answers\n- Be specific for better news results")

# Check configuration
if not API_KEYS_LOADED:
    st.error("⚠️ API Keys not configured. Please update the keys in app.py")
    st.stop()

if not llm or not app_graph:
    st.error("⚠️ System initialization failed. Check API keys and dependencies.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question or request news...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepare initial state
    initial_state = {
        "user_query": prompt,
        "news_category": news_category.lower(),
        "query_type": "",
        "news_results": "",
        "search_results": "",
        "final_response": "",
        "error_message": None
    }
    
    # Run workflow
    with st.spinner("NewsGenie is thinking..."):
        try:
            final_state = asyncio.run(app_graph.ainvoke(initial_state))
            response = final_state.get("final_response", "Sorry, an unexpected error occurred.")
        except Exception as e:
            st.error(f"Workflow Error: {e}")
            response = "Sorry, I encountered an error processing your request."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:** LangGraph + Groq + Streamlit")
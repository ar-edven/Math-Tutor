import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
_ = load_dotenv()

#Agent Schema
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("youtube api", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "youtube api", False: END}
        )
        graph.add_edge("youtube api", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
      
      
# Define the input schema
class YouTubeSearchInput(BaseModel):
    query: str = Field(..., description="Search query to find YouTube videos")

def convert_to_meaningful_string(youtube_data):
    items = youtube_data.get('items', [])
    meaningful_strings = []
    
    for item in items:
        title = item['snippet']['title']
        description = item['snippet']['description']
        channel_title = item['snippet']['channelTitle']
        published_at = item['snippet']['publishedAt']

        meaningful_string = f"Title: {title}\n" \
                            f"Description: {description}\n" \
                            f"Channel: {channel_title}\n" \
                            f"Published At: {published_at}\n"
                            
        if(item['id'].get('videoId')):
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            meaningful_string+=(f"Video URL: {video_url}\n")
            
        elif(item['id'].get('channelId')):
            channel_url = f"https://www.youtube.com/@{channel_title}"
            meaningful_string+=(f"Video URL: {channel_url}\n")
        elif(item['id'].get('playlistId')):
            playlist_id = item['id']['playlistId']
            playlist_url = f"https://www.youtube.com/watch?v=&list={playlist_id}"
            meaningful_string+=(f"Video URL: {playlist_url}\n")
        else:
            print("Bad response")
            
        print(meaningful_string)
                            

        meaningful_strings.append(meaningful_string)

    return "\n\n".join(meaningful_strings)

@tool(args_schema=YouTubeSearchInput)
def get_videos(query: str) -> dict:
    """Fetch YouTube video results based on a search query."""
    
    key = "AIzaSyBARxSlUgqVahXTCRRTGAqAWsOHDk7NGvY"
    part = "snippet"
    maxResults = 3
    
    url = f"https://www.googleapis.com/youtube/v3/search?key={key}&part={part}&q={query}&maxResults={maxResults}"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            videos = response.json()
            print(f"From getVideos: {videos}")
            return convert_to_meaningful_string(videos)
        else:
            raise Exception(f"API Request failed with status code: {response.status_code}")
      
    except requests.exceptions.RequestException as e:
        print("Error occurred during API Request", e)
        raise



# Example usage:
# result = get_videos("Python tutorial")
# print(result)

# print(get_videos.name)
# print(get_videos.description)
# print(get_videos.args)
# print("++++++++++++++++++++++\n\n")
# tool = get_videos
# print("<==========Tool Type=========>")
# print(type(tool))


# prompt = """You are a smart research assistant. Use the Youtube to look up information. \
# You are allowed to make multiple calls (either together or in sequence). \
# Only look up information when you are sure of what you want. \
# If you need to look up some information before asking a follow up question, you are allowed to do that!
# """

# model = ChatOpenAI(model="gpt-4o-mini")  #reduce inference cost
# abot = Agent(model, [tool], system=prompt)    

# print("<==GRAPH====MERMAID==>")
# print(abot.graph.get_graph().draw_mermaid())

# while(True):
#   message = input("Enter your message: ")
#   if(message.lower() == "exit"):
#     break
#   else:
#     messages = [HumanMessage(content=message)]
#     result = abot.graph.invoke({"messages": messages})
#     # print(result)
#     print(result['messages'][-1].content)




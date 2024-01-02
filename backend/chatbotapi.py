from pprint import pprint
from typing import Dict, List
from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
import openai
from langchain.chat_models import AzureChatOpenAI
import re  # regular expressions to parse the data
from typing import Dict
import inspect
import requests
import json
from memory import (
    get_chat_history,
    load_conversation_history,
    log_bot_message,
    log_user_message,
)
from fastapi import FastAPI, UploadFile, File, Request
from fastapi import BackgroundTasks

import pytz
from datetime import datetime
import math
import python_weather
import base64
import io

from slack_sdk import WebClient
import hmac
import hashlib
from starlette.responses import Response
# from azure.cosmos import CosmosClient, PartitionKey, exceptions
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.indexes import SearchIndexClient
# from azure.search.documents import SearchClient
# from azure.search.documents.indexes.models import(
#     ComplexField,
#     CorsOptions,
#     SearchIndex,
#     ScoringProfile,
#     SearchFieldDataType,
#     SimpleField,
#     SearchableField
#     )

# service_name = "mzd-aoai-svc3"
# admin_key = "service admin api key"

# index_name = "hotels-quickstart"

# # Cosmos DB configuration
# endpoint = os.getenv('COSMOS_ENDPOINT')
# key = os.getenv('COSMOS_KEY')
# database_name = os.getenv('COSMOS_DATABASE_NAME')
# container_name = os.getenv('COSMOS_CONTAINER_NAME')

# # Initialize the Cosmos client
# client = CosmosClient(endpoint, key)


load_dotenv()

# add bing search api key
bing_search_api_key = os.getenv('BING_SEARCH_API_KEY')

openai.api_key = os.getenv('OPENAI_API_KEY')


# Retrieve the environment variables using os.getenv
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
# OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
# OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
# deployment_id = OPENAI_DEPLOYMENT_NAME
# deployment_name = OPENAI_DEPLOYMENT_NAME

app = FastAPI()

# Slack WebClient for sending messages
slack_token = os.getenv('SLACK_TOKEN')
if not slack_token:
    raise ValueError("No SLACK_TOKEN found in environment variables")

slack_web_client = WebClient(token=slack_token)
print("Using Slack Token:", slack_token)

try:
    bot_id = slack_web_client.api_call("auth.test")['user_id']
except Exception as e:
    print(f"Error authenticating bot: {e}")
    raise e

# Function to verify Slack request signatures
def verify_slack_signature(request: Request, body: bytes) -> bool:
    timestamp = request.headers.get('X-Slack-Request-Timestamp')
    slack_signature = request.headers.get('X-Slack-Signature')
    signing_secret = os.getenv('SIGNING_SECRET')

    req = str.encode('v0:' + timestamp + ':') + body
    request_hash = 'v0=' + hmac.new(
        str.encode(signing_secret),
        req, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(request_hash, slack_signature)



json_file_path = "chat_histories/megazonecloud1.json"

# Dall E integration
def generate_image(prompt: str):
    try:
        # Make an API call to the Dall-E API
        response = openai.Image.create(prompt= prompt,model="dall-e-3", size="1024x1024",quality="standard", n=1)

        # The response contains a list of generated images
        image_url = response['data'][0]['url']
        return image_url
    except openai.error.OpenAIError as e:
        return f"Failed to generate image: {e}"


def calculator(num1, num2, operator):
    if operator == '+':
        return str(num1 + num2)
    elif operator == '-':
        return str(num1 - num2)
    elif operator == '*':
        return str(num1 * num2)
    elif operator == '/':
        return str(num1 / num2)
    elif operator == '**':
        return str(num1 ** num2)
    elif operator == 'sqrt':
        return str(math.sqrt(num1))
    else:
        return "Invalid operator"


def get_current_time(location):
    try:
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        # Get the current date in the timezone
        current_date = now.strftime("%Y-%m-%d")

        return f"Current time: {current_time}, Current date: {current_date}"
    except:
        return "Sorry, I couldn't find the timezone for that location."

# Function to delete the data from the chat history.
def reset_chathistory():
    with open(json_file_path, "w") as f:
        f.write("[]")
        print("Chat history reset successfully")




def create_booking(booking_subject, room_id, applicant_name, date, start_time, end_time, duration, attendees):
    url = "http://4.230.139.78:8080/create_booking"
    data = {
        "booking_subject": booking_subject,
        "room_id": room_id,
        "applicant_name": applicant_name,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "attendees": attendees
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return "Booking created successfully"
    else:
        return "Failed to create booking"
    

def read_booking(booking_id):
    url = f"http://4.230.139.78:8080/read_booking/{booking_id}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read booking"
    
def find_booking_by_time(date, start_time):
    url = "http://4.230.139.78:8080/read_booking_by_date_and_time"
    params = {"date": date, "start_time": start_time}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by date and time"

def find_booking_by_date(date):
    url = "http://4.230.139.78:8080/read_booking_by_date"
    params = {"date": date}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by date"

def find_booking_by_applicant_name(applicant_name):
    url = "http://4.230.139.78:8080/read_booking_by_applicant_name"
    params = {"applicant_name": applicant_name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by applicant name"

def find_booking_by_applicant_name_and_date(applicant_name,date):
    url = "http://4.230.139.78:8080/read_booking_by_applicant_name_and_date"
    params = {"applicant_name": applicant_name, "date": date}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by applicant name and date"

def read_all_bookings():
    url = "http://4.230.139.78:8080/read_all_bookings"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read all bookings"
    
def available_rooms(date, start_time, end_time, attendees):
    url = "http://4.230.139.78:8080/available_rooms"
    params = {"date": date, "start_time": start_time, "end_time": end_time, "attendees": attendees}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read available rooms"
    
def get_all_meeting_rooms():
    url = "http://4.230.139.78:8080/all_meeting_rooms"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read all meeting rooms"
    
def get_location_of_room(room_name: str):
    url = "http://4.230.139.78:8080/get_meeting_room_data"
    params = {"room_name": room_name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read the data of the meeting room"
    
def update_booking(booking_subject, new_booking_subject, room_id, applicant_name, date, new_booking_date,start_time, end_time, duration, attendees):
    url = "http://4.230.139.78:8080/update_booking"
    data = {
        "booking_subject": booking_subject,
        "new_booking_subject": new_booking_subject,
        "room_id": room_id,
        "applicant_name": applicant_name,
        "date": date,
        "new_booking_date": new_booking_date,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "attendees": attendees
        
    }
    response = requests.put(url, json=data)

    if response.status_code == 200:
        return "Booking updated successfully"
    else:
        return "Failed to update booking"


def get_coffee_menu():
    url = "http://4.230.139.78:8080/coffee_menu"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get coffee menu"

def get_coffee_orders():
    url = "http://4.230.139.78:8080/coffee_orders"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get coffee orders"
    

def order_coffee(customer_name, coffee_type, num_coffees, order_date, order_time, expected_time, pickup_status):
    url = "http://4.230.139.78:8080/order_coffee"
    data = {
        "customer_name": customer_name,
        "coffee_type": coffee_type,
        "num_coffees": num_coffees,
        "order_date": order_date,
        "order_time": order_time,
        "expected_time": expected_time,
        "pickup_status": pickup_status
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return "Coffee ordered successfully"
    else:
        return "Failed to order coffee"


def recommend_room_by_meeting_name(meeting_name: str):
    url = "http://4.230.139.78:8080/recommend_room_by_meeting_name"
    params = {"meeting_name": meeting_name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to recommend meeting room by meeting name"

def delete_booking_based_on_title_and_time(booking_subject, date, start_time):
    url = "http://4.230.139.78:8080/delete_booking"
    params = {"booking_subject": booking_subject, "date": date, "start_time": start_time}
    response = requests.delete(url, params=params)

    if response.status_code == 200:
        return "Booking deleted successfully"
    
def delete_booking_based_on_date_and_time(date, start_time):
    url = "http://4.230.139.78:8080/delete_booking_with_date_and_time"
    params = {"date": date, "start_time": start_time}
    response = requests.delete(url, params=params)

    if response.status_code == 200:
        return "Booking deleted successfully"

def bing_search(query: str) -> dict:
    headers = {"Ocp-Apim-Subscription-Key": bing_search_api_key}
    params = {"q": query, "count": 5, "offset": 0, "mkt": "en-US", "safesearch": "Moderate"}
    url = "https://api.bing.microsoft.com/v7.0/search"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get search results"


# Load the functions frm the functions.json file
with open("functions.json", "r") as f:
    functions = json.load(f)

# Extract the names of all functions
function_names = [f['name'] for f in functions]

# Print the function names
print(function_names)


# Define the available functions

available_functions = {
    "create_booking": create_booking,
    "read_booking": read_booking,
    "read_all_bookings": read_all_bookings,
    "update_booking": update_booking,
    "get_coffee_menu": get_coffee_menu,
    "order_coffee": order_coffee,
    "get_all_meeting_rooms": get_all_meeting_rooms,
    "get_coffee_orders": get_coffee_orders,
    "get_current_time": get_current_time,
    "calculator": calculator,
    "available_rooms": available_rooms,
    "recommend_by_meeting_name":recommend_room_by_meeting_name,
    "find_booking_by_time": find_booking_by_time,
    "delete_booking_based_on_title_and_time": delete_booking_based_on_title_and_time,
    "delete_booking_based_on_date_and_time":delete_booking_based_on_date_and_time,
    "find_booking_by_date": find_booking_by_date,
    "find_booking_by_applicant_name":find_booking_by_applicant_name,
    "find_booking_by_applicant_name_and_date": find_booking_by_applicant_name_and_date,
    "reset_chathistory": reset_chathistory,
    "get_location_of_room": get_location_of_room,
    "bing_search": bing_search,
    "generate_image":generate_image,
}   

# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True



openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_base = "https://api.openai.com"

def run_multiturn_conversation(messages, functions, available_functions):
    # Step 1: send the conversation and available functions to GPT

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        functions= functions,
        function_call="auto", 
        temperature=0.3 # 0.1, 0.5
    )

    # Booking meeting room : around 0.1 to 0.3
    # Ordering coffee : around 0.3 to 0.5

    # Step 2: check if GPT wanted to call a function
    while response["choices"][0]["finish_reason"] == 'function_call':
        response_message = response["choices"][0]["message"]
        print("Recommended Function call:")
        print(response_message.get("function_call"))
        print()
        
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        
        function_name = response_message["function_call"]["name"]
        
        # verify function exists
        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]  
        
        # verify function has correct number of arguments
        function_args = json.loads(response_message["function_call"]["arguments"])
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)
        # convert function_response to string
        function_response = str(function_response)
        
        print("Output of function call:")
        print(function_response)
        print()
        
        # Step 4: send the info on the function call and function response to GPT
        
        # adding assistant response to messages
        messages.append(
            {
                "role": response_message["role"],
                "function_call": {
                    "name": response_message["function_call"]["name"],
                    "arguments": response_message["function_call"]["arguments"],
                },
                "content": None
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("Messages in next request:")
        for message in messages:
            print(message)
        print()

        response = openai.ChatCompletion.create(
            model = "gpt-4-1106-preview",
            messages=messages,
            # deployment_id=deployment_name,
            function_call="auto",
            functions=functions,
            temperature=0
        )  # get a new response from GPT where it can see the function response

    return response

class ChatRequest(BaseModel):
    user_input: str


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r", encoding='utf-8') as f:
        prompt_template = f.read()

    return prompt_template



# Event handler for Slack messages
@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    print(f"Received event: {body}")  # Log the received event
    if not verify_slack_signature(request, body):
        return Response(status_code=403)
    # Add the event processing task to the background
    background_tasks.add_task(process_event, body)

    # Immediately respond with 200 OK
    return Response(status_code=200)


async def process_event(body):   
    event_payload = json.loads(body)
    event = event_payload.get("event", {})

    channel = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")

    # Check if the message is from the bot itself
    if user_id == bot_id:
        return {"challenge": event_payload.get("challenge")}  # No action if the message is from the bot itself

    # Fetch user information from Slack
    user_info_response = slack_web_client.users_info(user=user_id)
    if user_info_response.get("ok"):
        user_real_name = user_info_response["user"]["real_name"]

    else:
        user_real_name = "Unknown User"

    history_file = load_conversation_history("chatsession"+ user_real_name)
    chat_history = get_chat_history("chatsession"+ user_real_name)
    messages = [
            {
                "role": "system",
                #"content": next_prompt
                "content": "Here is your chat history:\n" + chat_history
            },
            {
                "role": "system",
                "content": f"As an AI system, your core function revolves around precise time management and user-specific interactions. Utilize the ‘get_current_time’ function with the Asia/Seoul timezone setting for all date and time references. Address the user consistently as {user_real_name}. Employ Bing Search and Dall-E as your primary tool for internet queries and generating images, especially when directed by the user. Prioritize user confirmation before proceeding with scheduling meetings or executing tasks. In scenarios lacking specific date details, default to the current date as per Seoul time. Your responses should build upon previous interactions with the user, maintaining relevance and coherence in the conversation flow."
            },
            {
                "role": "system",
                "content": f"Your role as an AI assistant places you at the heart of supporting {user_real_name} at the Microsoft Technology Center, MegaZone Cloud, in Seoul. Your assistance is centered around Azure-related tasks and queries. Communication should be exclusively in Korean, embedding cultural and linguistic nuances. Each response you provide must be accompanied by a clear rationale, outlining the thought process and methods used to reach your conclusions. This approach not only informs but also educates {user_real_name} about the reasoning behind each answer.",
            },
            {
                "role": "user",
                "content": text
            }
        ]
    response = run_multiturn_conversation(messages, functions, available_functions)
    log_user_message(history_file, text)

    # return {"bot_response": response["output"]}
    if "choices" in response and response["choices"]:
        response_text = response["choices"][0]["message"]["content"]
        print("Response from GPT:", response_text)
        log_bot_message(history_file, response_text)
    else:
        response_text = "Sorry, I didn't get that."
        print("No choices found in response or an error occurred.")
    
    # Send the GPT response back to the Slack channel
    try:
        slack_web_client.chat_postMessage(
            channel=channel,  # Respond in the same channel as the message
            text=response_text
            )
    except Exception as e:
        print(f"Error posting message: {e}")
    
    return {"challenge": event_payload.get("challenge")}

@app.post("/chat_response/{conversation_id}")
async def get_bot_response(req: ChatRequest, conversation_id: str) -> Dict[str, str]:
    history_file = load_conversation_history(conversation_id)
    chat_history = get_chat_history(conversation_id)
    # Construct the next prompt based on the user input and the chat history
    #next_prompt = construct_next_prompt(req.user_input, chat_history)
    messages = [
            {
                "role": "system",
                #"content": next_prompt
                "content": "Here is your chat history:\n" + chat_history
            },
            {
                "role": "system",
                "content": "You must always check the today date by calling the function, get_current_time of Asia/Seoul to check the current date and time. Always use the bing_search function to search the question from the user if the user asks to search on internet via bing search and always think that the user name is 양동준. Always seek and obtain final confirmation from the user before proceeding to book a meeting or executing any action. Always think the date is today date if the user doesn't give any specific dates. You always call the function, get_current_time of Asia/Seoul to check the current date and time before booking the meeting room according to user requirements and you must always reference the previous conversation with the user to ensure a coherent and relevant response for the next interaction." },
            {
                "role": "system",
                "content": """  The user has requested to book a meeting room at a time when our standard rooms
                                are fully booked. Explore alternative options such as recommending available smaller rooms where extra
                                chairs can be added if the number of attendees is slightly higher than the room's capacity or suggesting the available room at a different time that aligns with the user's schedule, for example searching the other available time on that day. When providing alternatives, explain to the user why each option is suitable. For example,
                                if recommending a room in a different building, mention the amenities and proximity. If 
                                suggesting adding chairs to a smaller room, reassure the user that comfort and space will still
                                be adequate. Only when the user requests to book the meeting with the title or name or subject of the meeting and also asks to book like the last time, you must always recommend the meeting booking info with the time, number of participants and the meeting room based on that meeting name. Then if the user confirms that recommendation, then proceed to book the meeting with those information. The assistant should only answer questions related to its assigned task. If asked about other things, the assistant should respond with 'I am sorry, because it's not my assigned task so I can't help you.' This is to prevent hallucinations and prompt injection.
                                If the user asks about the location of the meeting, answers in details of the booked meeting room including the location. If the user talks about the cancellation of the meeting,always ask whether they would like to cancel that meeting booking or not in a professional and accommodating tone. Always make sure to ask and confirm the user decision before deleting meeting booking.
                                Always recommend ordering hot drink or coffee because the weather is getting colder these days whenever after the meeting room booking is completed. 
                                """ },
                                
           
            {
                "role": "system",
                "content": "You are an AI assistant that helps with meeting room bookings and coffee orders for the user named 양동준 who is working at MegaZone Cloud company which is located in Seoul, South Korea and always gives reasons for your answers and how did you get those answers. Always use Korean language to communicate with the user.",
            },
            {
                "role": "user",
                "content": req.user_input
            }
        ]
    
    response = run_multiturn_conversation(messages, functions, available_functions)
    log_user_message(history_file, req.user_input)
    log_bot_message(history_file, str(response))

    # return {"bot_response": response["output"]}
    print("Response from GPT:")
    print(response)
    if "choices" in response and response["choices"]:
        print(response["choices"][0]["message"]["content"])
    else:
        print("No choices found in response.")
    # Step 5: return the response from GPT
    return {"bot_response": response["choices"][0]["message"]["content"]}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

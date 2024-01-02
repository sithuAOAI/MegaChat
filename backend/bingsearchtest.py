import requests

def test_bing_search_api():
    # Replace 'your_api_key_here' with your actual Bing Search API key
   
    endpoint = 'https://api.bing.microsoft.com/v7.0/search'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    params = {'q': 'Microsoft', 'count': 10, 'offset': 0, 'mkt': 'en-US'}
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        
        # If the request is successful, print a summary of results
        print("The Bing Search API is working properly.")
        print(f"HTTP Status Code: {response.status_code}")
        json_response = response.json()
        print(f"Number of results: {json_response['webPages']['totalEstimatedMatches']}")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
    except requests.exceptions.ConnectionError as err:
        print(f"Error Connecting: {err}")
    except requests.exceptions.Timeout as err:
        print(f"Timeout Error: {err}")
    except requests.exceptions.RequestException as err:
        print(f"An Error occurred: {err}")

# Remember to replace 'your_api_key_here' with your actual API key before running the function
test_bing_search_api()

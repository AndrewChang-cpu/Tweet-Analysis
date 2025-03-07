import json

# Load the JSON file containing multiple tweets
tweets = []
with open("../data/data1.json", "r", encoding="utf-8") as f:
    for line in f:
        tweets.append(json.loads(line.strip()))


def extract_urls(obj, collected_urls):
    """
    Recursively searches for any key containing 'url' in a nested dictionary or list
    and adds its value to collected_urls.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "url" in key and isinstance(value, str):
                collected_urls.append(value)
            extract_urls(value, collected_urls)
    elif isinstance(obj, list):
        for item in obj:
            extract_urls(item, collected_urls)


# Extract all URLs from every field
all_urls = []
for tweet in tweets:
    extract_urls(tweet, all_urls)
# print(all_urls)

with open('urls.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(all_urls))
    
    
# Extract hashtags
all_hashtags = [ht["text"] for tweet in tweets for ht in tweet.get("entities", {}).get("hashtags", [])]
# print(all_hashtags)

with open('hashtags.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(all_hashtags))

from gradio_client import Client, handle_file
import time
import sys

client = Client("http://127.0.0.1:7861")

print("Triggering render...")
# The API expects a file object for the JSON
result = client.predict(
		json_file=handle_file("/Users/Star/chatterbox/test_render.json"),
		voice="young British Woman",
		api_name="/render_movie"
)
print(f"Render result: {result}")

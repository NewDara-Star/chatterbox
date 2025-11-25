from gradio_client import Client
import json

client = Client("http://127.0.0.1:7861")
info = client.view_api(return_format="dict")
print(json.dumps(info, indent=2))

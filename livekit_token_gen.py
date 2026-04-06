from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('LIVEKIT_URL')
api_key = os.getenv('LIVEKIT_API_KEY')
api_secret = os.getenv('LIVEKIT_API_SECRET')
room_name = 'soulx-flashhead-room'

token = api.AccessToken(api_key, api_secret) \
    .with_identity('viewer-1') \
    .with_name('Human Viewer') \
    .with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,    # so your mic works
        can_subscribe=True,  # so you can see/hear the bot
    )) \
    .to_jwt()

print("Token:", token)
print("URL:", url)
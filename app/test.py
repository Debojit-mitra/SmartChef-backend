# test_chat.py

import asyncio
import websockets
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatTester:
    def __init__(self, websocket_url="ws://localhost:8000/ws/chat"):
        self.websocket_url = websocket_url
        self.chat_id = None
        self.websocket = None
        self.current_response = []  # Store chunks of current response

    async def connect(self):
        """Connect to websocket server"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("Connected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False

    async def send_message(self, message):
        """Send a message to the server"""
        if not self.websocket:
            logger.error("Not connected to server")
            return

        try:
            await self.websocket.send(json.dumps({
                "message": message,
                "chat_id": self.chat_id
            }))

            # Reset current response
            self.current_response = []
            first_chunk = True

            # Process responses
            while True:
                response = await self.websocket.recv()
                data = json.loads(response)
                
                if data["type"] == "chat_id":
                    self.chat_id = data["chat_id"]
                    logger.info(f"Chat ID: {self.chat_id}")
                
                elif data["type"] == "content":
                    if first_chunk:
                        print("Assistant: ", end="", flush=True)
                        first_chunk = False
                    print(data['content'], end="", flush=True)
                    self.current_response.append(data['content'])
                
                elif data["type"] == "error":
                    logger.error(f"Error: {data['error']}")
                    break
                
                elif data["type"] == "done":
                    print("\n")  # New line after completion
                    logger.info(f"Processing time: {data['query_time']}s")
                    break
                
                elif data["type"] == "heartbeat":
                    logger.debug("Heartbeat received")

        except Exception as e:
            logger.error(f"Error sending/receiving message: {str(e)}")

    async def close(self):
        """Close the websocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed")

async def main():
    tester = ChatTester()
    
    if not await tester.connect():
        return

    try:
        print("\nChat Test Started (type 'quit' to exit)")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print()  # New line for readability
            await tester.send_message(user_input)

    finally:
        await tester.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest ended by user")
    except Exception as e:
        print(f"Test ended with error: {str(e)}")
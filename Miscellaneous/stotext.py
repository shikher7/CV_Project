import pyaudio
import websockets
import asyncio
import base64
import json
import logging
import threading

class SpeechToText:
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
    AUTH_KEY = "key"  # Insert your AssemblyAI key here

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.text = ""
        self.is_speaking = False
        self._ws = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.buffered_text = []
        self.final_text = ""
        self.empty_count = 0  # Add this line


    def start_recording(self):
        self.is_speaking = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER
        )

        self.loop.run_until_complete(self.send_receive())

    def stop_recording(self):
        self.is_speaking = False
        self.stream.stop_stream()
        self.stream.close()

    async def send_receive(self):
        print(f'Connecting websocket to url {self.URL}')
        async with websockets.connect(self.URL, extra_headers=(("Authorization", self.AUTH_KEY),),
                                      ping_interval=5, ping_timeout=20) as _ws:
            self._ws = _ws
            await asyncio.sleep(0.1)
            print("Receiving SessionBegins ...")
            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages ...")

            tasks = [self.loop.create_task(self.send()), self.loop.create_task(self.receive())]
            await asyncio.wait(tasks)

    async def send(self):
        while self.is_speaking:
            try:
                data = self.stream.read(self.FRAMES_PER_BUFFER)
                data = base64.b64encode(data).decode("utf-8")
                json_data = json.dumps({"audio_data": str(data)})
                await self._ws.send(json_data)
            except websockets.exceptions.ConnectionClosedError as e:
                logging.exception(f'Websocket closed {e.code}')
                break
            except Exception:
                logging.exception('Not a websocket')
            await asyncio.sleep(0.01)

    async def receive(self):
        while self.is_speaking:
            try:
                result_str = await self._ws.recv()
                result_json = json.loads(result_str)
                if 'text' in result_json and result_json['text'] != "":
                    self.buffered_text.append(result_json['text'])
                    self.final_text = result_json['text']
                    self.empty_count = 0
                elif 'text' in result_json and result_json['text'] == "":
                    # print(self.empty_count)
                    self.empty_count += 1
                    if self.empty_count >= 5:
                        self.process_buffered_text()
                        self.is_speaking = False
                print(self.final_text)
            except websockets.exceptions.ConnectionClosedError as e:
                logging.exception(f'Websocket closed {e.code}')
                break
            except Exception:
                logging.exception('Not a websocket')

    def process_buffered_text(self):
        # This function processes buffered text
        # For now, we only get the last result
        if self.buffered_text:
            self.text = self.final_text
            self.buffered_text = []
            self.final_text = ""

    def transcribe(self):
        return self.text

def main():
    speech_to_text = SpeechToText()
    recording_thread = None

    while True:
        command = input("Enter a command (start, stop, print, quit): ")

        if command == "start":
            recording_thread = threading.Thread(target=speech_to_text.start_recording)
            recording_thread.start()
        elif command == "stop":
            if recording_thread is not None:
                speech_to_text.stop_recording()
                recording_thread.join()
                recording_thread = None
        elif command == "print":
            print(speech_to_text.transcribe())
        elif command == "quit":
            if recording_thread is not None:
                speech_to_text.stop_recording()
                recording_thread.join()
            break
        else:
            print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()

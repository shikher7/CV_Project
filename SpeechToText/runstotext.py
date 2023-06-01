import time
from threading import Thread
from stotext import SpeechToText

def use_speech_to_text(speech_to_text):
    speech_to_text.start_recording()

    time.sleep(1000000)
    speech_to_text.stop_recording()

    text = speech_to_text.get_text()

    print(text)

    speech_to_text.reset()


def main():
    speech_to_text = SpeechToText()

    thread = Thread(target=use_speech_to_text, args=(speech_to_text,))
    thread.start()

    thread.join()


if __name__ == "__main__":
    main()
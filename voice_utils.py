"""
================================================================================
NeuralStride AI - Voice Engine (Text-to-Speech)
================================================================================

Provides real-time voice feedback during workouts using Windows SAPI TTS.

HOW IT WORKS:
    Uses a background thread with a message queue so that voice output
    doesn't block the main camera/pose processing loop. When speak() is
    called, the message is placed in a queue and the worker thread picks
    it up asynchronously.

VOICE QUEUE FLOW:
    1. exercise logic calls voice.speak("Good squat")
    2. message is placed in a thread-safe queue (queue.Queue)
    3. worker thread (running in background) picks up the message
    4. Windows SAPI SpVoice synthesizes and plays the audio
    5. If multiple messages arrive, they play sequentially (no overlap)

WHY A SEPARATE THREAD:
    Without a thread, speaker.Speak() would block the main loop for 1-2
    seconds per message, freezing the camera feed and losing frames.
    With the thread, voice plays in the background while the loop continues.

WINDOWS ONLY:
    Uses Windows COM (pythoncom + win32com.client) for SAPI SpVoice.
    pythoncom.CoInitialize() is required because COM objects must run
    in a thread that has initialized the COM library.

CLEANUP:
    stop() sends a None sentinel to the queue, signaling the worker thread
    to exit. The thread is joined with a 1-second timeout to avoid hanging.

DEPENDENCIES:
    pip install pywin32
================================================================================
"""

import queue
import threading
import pythoncom
import win32com.client


class VoiceEngine:
    """
    Async text-to-speech engine using Windows SAPI.
    Voice messages are queued and played in a background thread
    so the main camera loop never blocks on speech output.
    """

    def __init__(self):
        """
        Creates the message queue and starts the background worker thread.
        The thread runs daemon mode so it exits automatically if the
        program crashes (no orphan threads).
        """
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """
        Background thread that processes voice messages from the queue.
        Runs in an infinite loop until it receives a None sentinel message
        (sent by stop()), which breaks the loop and cleans up COM.

        Flow per message:
            1. Block on queue.get() until a message arrives
            2. Check if it's the None sentinel -> exit loop
            3. Print to console for debugging ("Speaking: ...")
            4. Call speaker.Speak() to play the voice audio
            5. Mark task as done in the queue
        """
        # COM must be initialized in each thread that uses COM objects
        pythoncom.CoInitialize()

        # Create the Windows SAPI voice synthesizer
        speaker = win32com.client.Dispatch("SAPI.SpVoice")

        while True:
            # Block until a message is available in the queue
            message = self.queue.get()

            # None is the sentinel value — stop() sends this to end the thread
            if message is None:
                break

            try:
                # Print what's being spoken (useful for debugging without speakers)
                print(f"Speaking: {message}", flush=True)
                # SAPI.Speak() is synchronous — it blocks this thread until
                # the audio finishes playing, but that's fine because this IS
                # the background thread. The main loop is unaffected.
                speaker.Speak(message)
            except Exception as e:
                print(f"Voice error: {e}", flush=True)

            # Mark this queue item as processed
            self.queue.task_done()

        # Clean up COM library before thread exits
        pythoncom.CoUninitialize()

    def speak(self, message):
        """
        Queue a voice message for playback. Non-blocking — returns immediately.
        Clears any stale messages from the queue before adding the new one,
        so only the most recent feedback is spoken (no backlog delays).
        """
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        self.queue.put(message)

    def stop(self):
        """
        Signal the worker thread to stop and wait for it to finish.
        Called when the workout ends (after 'q' is pressed in main.py).
        Sends None as a sentinel — the worker thread breaks on receiving it.
        1-second timeout prevents hanging if something goes wrong.
        """
        self.queue.put(None)
        self.thread.join(timeout=1)

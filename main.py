"""
main.py
The integrated bidirectional sign language translation system.

Presents a menu and runs either pipeline:
    1. Sign Language -> Speech   (Pipeline A, webcam + LSTM + text-to-speech)
    2. Speech -> Sign Language   (Pipeline B, mic + sign video playback)

Both pipelines live in their own files (pipeline_a.py, pipeline_b.py); this
file just ties them together behind one menu so the whole system runs as a
single application.

"""

import pipeline_a
import pipeline_b


BANNER = r"""
============================================================
  BIDIRECTIONAL SIGN LANGUAGE TRANSLATION SYSTEM
============================================================
  1 . Sign Language  ->  Speech     (use your webcam)
  2 . Speech         ->  Sign        (use your microphone)
  q . Quit
------------------------------------------------------------
"""


def main():
    while True:
        print(BANNER)
        choice = input("Choose an option (1 / 2 / q): ").strip().lower()

        if choice == "1":
            print("\nStarting Sign-to-Speech. Press q in the webcam window to return.\n")
            pipeline_a.main()
        elif choice == "2":
            print("\nStarting Speech-to-Sign. Type q at its prompt to return.\n")
            pipeline_b.main()
        elif choice == "q":
            print("Goodbye.")
            break
        else:
            print("Please enter 1, 2, or q.")


if __name__ == "__main__":
    main()
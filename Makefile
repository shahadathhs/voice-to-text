# Default values
AUDIO ?= audio.wav

MODEL ?= base

.PHONY: build run translate diarize clean

# Build the Docker image
build:
	sudo docker compose build

# Basic transcription
run:
	sudo docker compose run --rm whisper $(AUDIO) --model $(MODEL)

# Transcription + Translation
translate:
	sudo docker compose run --rm whisper $(AUDIO) --model $(MODEL) --translate

# Transcription + Diarization
diarize:
	sudo docker compose run --rm whisper $(AUDIO) --model $(MODEL) --diarize

# All-in-one: Transcribe + Translate + Diarize
all:
	sudo docker compose run --rm whisper $(AUDIO) --model $(MODEL) --translate --diarize

# Clean transcripts folder
clean:
	rm -rf transcripts/*

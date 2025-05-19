import os
import tempfile
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import speech_recognition as sr
from PIL import Image
import streamlit as st
from src.document_processor import DocumentProcessor
from src.embedding_utils import VectorStoreManager
from src.exception_handler import handle_exception

def process_uploaded_video(video_path):
    try:
        print(f"Starting video processing for: {video_path}")
        tmp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {tmp_dir}")
        audio_path = os.path.join(tmp_dir, "audio.wav")
        
        # Initialize UI status message
        status = st.empty()
        status.info("Processing video: Analyzing file...")
        
        # Extract frames
        print("Extracting video information...")
        clip = VideoFileClip(video_path)
        print(f"Video loaded. Duration: {clip.duration}s, FPS: {clip.fps}, Resolution: {clip.size}")
        
        status.info("Processing video: Extracting frames every 3 seconds...")
        print("Extracting frames every 3 seconds...")
        os.makedirs("figures", exist_ok=True)
        frame_count = 0
        
        for t in range(0, int(clip.duration), 3):
            print(f"Extracting frame at {t} seconds...")
            frame = clip.get_frame(t)
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join("figures", f"frame_{t}.jpg")
            frame_img.save(frame_path)
            print(f"Saved frame to {frame_path}")
            frame_count += 1
            
            # Update UI every 5 frames
            if frame_count % 5 == 0:
                status.info(f"Processing video: Extracted {frame_count} frames so far...")
        
        print(f"Frame extraction complete. Saved {frame_count} frames to figures/")
        
        # Extract and process audio
        status.info("Processing video: Extracting audio track...")
        print("Extracting audio from video...")
        clip.audio.write_audiofile(audio_path)
        print(f"Audio extracted to {audio_path}")
        
        # Initialize speech recognition
        status.info("Processing video: Preparing for transcription...")
        print("Initializing speech recognition...")
        recognizer = sr.Recognizer()
        
        # Load audio and split into chunks
        print("Loading audio file for transcription...")
        audio = AudioSegment.from_wav(audio_path)
        print(f"Audio duration: {len(audio)/1000}s")
        
        # Process audio in 60-second chunks
        print("Splitting audio into 60-second chunks...")
        audio_chunks = audio[::60000]  # Split every 60 seconds
        print(f"Created {len(audio_chunks)} audio chunks")
        
        # Transcribe audio
        transcriptions = []
        status.info(f"Processing video: Transcribing audio ({len(audio_chunks)} chunks)...")
        
        for i, chunk in enumerate(audio_chunks):
            chunk_status_msg = f"Processing video: Transcribing audio chunk {i+1}/{len(audio_chunks)}..."
            status.info(chunk_status_msg)
            print(f"Processing audio chunk {i+1}/{len(audio_chunks)}...")
            
            chunk_path = os.path.join(tmp_dir, f"chunk_{i}.wav")
            print(f"Exporting chunk to {chunk_path}")
            chunk.export(chunk_path, format='wav')
            
            with sr.AudioFile(chunk_path) as source:
                print(f"Reading audio data from chunk {i+1}...")
                audio_data = recognizer.record(source)
                try:
                    print(f"Transcribing chunk {i+1} using Google Speech Recognition...")
                    text = recognizer.recognize_google(audio_data)
                    print(f"Transcription successful: {text[:60]}...")
                    transcriptions.append(text)
                    
                except Exception as e:
                    error_msg = f"Failed to transcribe chunk {i+1}: {str(e)}"
                    print(f"WARNING: {error_msg}")
                    transcriptions.append("[Unrecognized audio]")
        
        # Combine all transcriptions
        transcribed_text = " ".join(transcriptions)
        print(f"Transcription complete. Total text length: {len(transcribed_text)} characters")
        print(f"Transcription sample: {transcribed_text[:200]}...")
        
        # Process and summarize content
        status.info("Processing video: Summarizing content...")
        print("Initializing DocumentProcessor for content summarization...")
        processor = DocumentProcessor()
        
        # Process empty elements list but add transcription as text element
        print("Creating processed document structure...")
        processed = processor._process_elements([])
        processed["text_elements"] = [transcribed_text]
        
        print("Summarizing transcribed text...")
        status.info("Processing video: Generating text summaries...")
        processed["text_summaries"] = processor._summarize_texts([transcribed_text])
        
        print("Processing extracted frames...")
        status.info("Processing video: Analyzing extracted frames...")
        processed["image_paths"] = [f for f in os.listdir("figures") if f.startswith("frame_")]
        print(f"Found {len(processed['image_paths'])} frame images to analyze")
        
        print("Generating image summaries...")
        status.info(f"Processing video: Analyzing {len(processed['image_paths'])} frames...")
        processed["image_summaries"] = processor._summarize_images(processed["image_paths"])
        
        # Create vector store
        filename_base = os.path.splitext(os.path.basename(video_path))[0]
        store_name = f"{filename_base}_index_faiss"
        
        print(f"Creating vector store '{store_name}'...")
        status.info("Processing video: Creating searchable database...")
        vectorstore, embeddings = VectorStoreManager.create_vector_store(processed, store_name)
        
        total_entries = len(processed["text_summaries"]) + len(processed["image_summaries"])
        success_msg = f"Video processing complete: Created database with {total_entries} entries"
        print(success_msg)
        status.success(success_msg)
        status.empty()
        
        return vectorstore, embeddings, store_name
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        print(f"ERROR: {error_msg}")
        status = st.empty()
        status.error(error_msg)
        handle_exception(e)
        return None, None, None
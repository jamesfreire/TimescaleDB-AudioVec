#!/usr/bin/env python3
"""
Script to generate audio embeddings using VGGish and store them in TimescaleDB.

This script extends the original VGGish inference demo to:
1. Process multiple audio files
2. Store the embeddings in a TimescaleDB vector database
3. Include metadata like timestamp, audio filename, and embedding position

Requirements:
- TensorFlow (compatible with both TF 1.x and 2.x)
- NumPy
- psycopg2 (for PostgreSQL/TimescaleDB connection)
- VGGish model files (checkpoint and PCA params)
"""

from __future__ import print_function

import os
import sys
import argparse
import datetime
import numpy as np
from scipy.io import wavfile
import six
import psycopg2
from psycopg2.extras import execute_values
import json

# Import VGGish modules
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

# Check TensorFlow version and import appropriate modules
import tensorflow as tf
print(f"Using TensorFlow version: {tf.__version__}")
is_tf2 = tf.__version__.startswith('2')

if is_tf2:
    # TensorFlow 2.x compatibility
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
    print("Running in TensorFlow 2.x compatibility mode")
else:
    # TensorFlow 1.x
    import tensorflow as tf

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate audio embeddings and store in TimescaleDB')
    
    parser.add_argument('--wav_dir', type=str, default=None,
                        help='Directory containing WAV files to process')
    parser.add_argument('--wav_file', type=str, default=None,
                        help='Single WAV file to process')
    parser.add_argument('--checkpoint', type=str, default='vggish_model.ckpt',
                        help='Path to the VGGish checkpoint file')
    parser.add_argument('--pca_params', type=str, default='vggish_pca_params.npz',
                        help='Path to the VGGish PCA parameters file')
    parser.add_argument('--db_name', type=str, required=True,
                        help='TimescaleDB database name')
    parser.add_argument('--db_user', type=str, required=True,
                        help='TimescaleDB username')
    parser.add_argument('--db_password', type=str, required=True,
                        help='TimescaleDB password')
    parser.add_argument('--db_host', type=str, default='localhost',
                        help='TimescaleDB host')
    parser.add_argument('--db_port', type=int, default=5432,
                        help='TimescaleDB port')
    parser.add_argument('--table_name', type=str, default='audio_embeddings',
                        help='Table name for storing embeddings')
    
    return parser.parse_args()

def setup_database(conn, table_name):
    """Create the necessary tables and extensions in TimescaleDB."""
    with conn.cursor() as cur:
        # Enable the vector extension if not already enabled
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE;")
        
        # Create the table for audio embeddings
        # IMPORTANT: Make timestamp part of the PRIMARY KEY to satisfy TimescaleDB requirements
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TIMESTAMPTZ NOT NULL,
            audio_file TEXT NOT NULL,
            segment_id INTEGER NOT NULL,
            embedding vector(128),
            metadata JSONB,
            PRIMARY KEY (timestamp, audio_file, segment_id)
        );
        """)
        
        # Create a hypertable partitioned by time
        try:
            cur.execute(f"""
            SELECT create_hypertable('{table_name}', 'timestamp', 
                                    if_not_exists => TRUE);
            """)
        except Exception as e:
            # If the table was already a hypertable, this is fine
            print(f"Note: {e}")
            conn.rollback()
        
        # Create an index on the vector column for similarity search
        cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
        ON {table_name} USING diskann (embedding vector_cosine_ops);
        """)
        
        conn.commit()
        print("Database Setup Complete")

def process_audio_file(wav_file, sess, features_tensor, embedding_tensor, pproc):
    """Process a single audio file and return embeddings with metadata."""
    try:
        # Extract the embeddings
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)
        
        # Get the audio file name
        if isinstance(wav_file, str):
            audio_file_name = os.path.basename(wav_file)
        else:
            audio_file_name = "synthetic_audio.wav"
        
        # Create timestamp (current time for the entire file)
        timestamp = datetime.datetime.now()
        
        # Prepare results with metadata
        results = []
        for i, embedding in enumerate(postprocessed_batch):
            # Each embedding represents roughly 1 second of audio
            segment_timestamp = timestamp + datetime.timedelta(seconds=i)
            
            # Basic metadata about this segment
            metadata = {
                "segment_duration_ms": 960,  # 96 frames * 10ms
                "segment_start_time_s": i * 0.96,
                "embedding_dimension": 128
            }
            
            results.append({
                "timestamp": segment_timestamp,
                "audio_file": audio_file_name,
                "segment_id": i,
                "embedding": embedding.tolist(),
                "metadata": metadata
            })
        
        return results
    
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return []

def store_embeddings(conn, table_name, embeddings_data):
    """Store the embeddings in TimescaleDB."""
    with conn.cursor() as cur:
        # Prepare the data for bulk insert
        values = [
            (
                item["timestamp"],
                item["audio_file"],
                item["segment_id"],
                item["embedding"],
                json.dumps(item["metadata"])
            )
            for item in embeddings_data
        ]
        print("inserting into db")    
        # Use execute_values for efficient bulk insertion
        # Include ON CONFLICT clause to handle potential duplicate timestamps
        execute_values(
            cur,
            f"""
            INSERT INTO {table_name} 
            (timestamp, audio_file, segment_id, embedding, metadata)
            VALUES %s
            ON CONFLICT (timestamp, audio_file, segment_id) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
            """,
            values,
            template="(%s, %s, %s, %s::vector, %s)"
        )
        
        conn.commit()

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not args.wav_file and not args.wav_dir:
        # Generate synthetic audio if no input is provided
        num_secs = 5
        freq = 1000
        sr = 44100
        t = np.linspace(0, num_secs, int(num_secs * sr))
        x = np.sin(2 * np.pi * freq * t)
        samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
        wav_file = six.BytesIO()
        wavfile.write(wav_file, sr, samples)
        wav_file.seek(0)
        wav_files = [wav_file]
    elif args.wav_file:
        wav_files = [args.wav_file]
    else:
        # Get all WAV files from the directory
        wav_files = [
            os.path.join(args.wav_dir, f) 
            for f in os.listdir(args.wav_dir) 
            if f.lower().endswith('.wav')
        ]
    
    # Prepare the model and processor
    pproc = vggish_postprocess.Postprocessor(args.pca_params)
    
    # Connect to the database
    db_conn = psycopg2.connect(
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port
    )
    
    # Set up the database table
    setup_database(db_conn, args.table_name)
    
    # Initialize TensorFlow session
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode and load the checkpoint
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, args.checkpoint)
        
        # Get input and output tensors
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        
        # Process each WAV file
        for wav_file in wav_files:
            print(f"Processing: {wav_file}")
            embeddings_data = process_audio_file(
                wav_file, sess, features_tensor, embedding_tensor, pproc)
            
            if embeddings_data:
                print(f"Generated {len(embeddings_data)} embeddings")
                store_embeddings(db_conn, args.table_name, embeddings_data)
                print(f"Stored embeddings in TimescaleDB")
            else:
                print(f"No embeddings generated for {wav_file}")
    
    # Close the database connection
    db_conn.close()
    print("Processing complete.")

if __name__ == '__main__':
    main()

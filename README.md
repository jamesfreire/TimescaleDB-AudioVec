# AudioVec - Audio Embedding Vector Database with TimescaleDB 

AudioVec is an exerimentation on storing and searching audio embeddings that are generated via the [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) and [TimescaleDB's vector database capabilities](https://github.com/timescale/pgvectorscale). My goal was for generating an embeddings to ultimately gain more experience in working with vectors in TimescaleDB. With TimescaleDB you can analyze audio content, find similar audio segments, and even build an audio-based retrieval system. 

This script extends the original [VGGish inference demo.](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)


## Features

- Generate 128-dimensional audio embeddings from WAV files using Google's VGGish model
- Store embeddings in TimescaleDB with vector search capabilities
- Process individual files or entire directories of audio
- Perform similarity searches to find matching audio content
- Includes time-series capabilities for temporal queries
- Built on industry-standard tools: TensorFlow, PostgreSQL, and TimescaleDB

## Requirements

- Python 3.7+
- TensorFlow 2.x
- VGGish - [Details on how to setup VGGish is here](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
- Psycopg2
- PostgreSQL 13+ with TimescaleDB 2.0+ extension
- TimescaleDB Vector extension

## Quick Start

### 1. Installation

```bash
# Clone this repository
git clone https://github.com/jamesfreire/audiovec.git
cd audiovec

# Install dependencies
pip install tensorflow tensorflow-hub numpy scipy psycopg2-binary
```


##### VGGish depends on the following Python packages:

* [`numpy`](http://www.numpy.org/)
* [`resampy`](http://resampy.readthedocs.io/en/latest/)
* [`tensorflow`](http://www.tensorflow.org/)
* [`tf_slim`](https://github.com/google-research/tf-slim)
* [`six`](https://pythonhosted.org/six/)
* [`soundfile`](https://pysoundfile.readthedocs.io/)

These are all easily installable via, e.g., `pip install numpy` (as in the
sample installation session below). Any reasonably recent version of these
packages should work.

##### VGGish also requires downloading two data files:

* [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt),
  in TensorFlow checkpoint format.
* [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz),
  in NumPy compressed archive format.

### 2. Database Setup

[Instructions on how to install TimescaleDB is available on their site](https://docs.timescale.com/self-hosted/latest/install/)


The code will automatically install the vector extension and setup the audio embeddings table, here called audio_embeddings, or whatever you define at the command line. If you wish to set it up manually you can execute:
```bash
psql -U postgres -c "CREATE DATABASE audio_vectors;"
psql -U postgres -d audio_vectors -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
psql -U postgres -d audio_vectors -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Run the Embedding Generator

```bash
# Process a directory of audio files
python vggish_to_timescaledb.py \ 
                 --wav_dir /path/to/audio \
                 --checkpoint /path/to/custom/vggish_model.ckpt \
                 --pca_params /path/to/custom/vggish_pca_params.npz \
                 --db_name audio_embeddings\ 
                 --db_user postgres \
                 --db_password your_password
```

 You can also process an individual wav file using:
```bash
--wav-file /path/to/audio.wav
```

The code will also generate synthetic audio if no input is provided.

## How It Works

AudioVec uses Google's VGGish neural network to convert audio into a consistent "embedding" representation - a 128-dimensional vector that captures the audio's characteristics. These embeddings are then stored in TimescaleDB with the vector extension, which enables fast similarity searches.

### Embedding Generation Process:

1. Audio files are loaded and segmented into ~1 second chunks
2. Each segment is processed through the VGGish model
3. The resulting 128-dimensional embeddings are stored with metadata
4. TimescaleDB organizes the data for efficient retrieval

### Similarity Search:

The vector extension in TimescaleDB uses [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to find audio segments that match a query embedding. This allows you to find:
- Similar sounds to a reference audio
- Repeated patterns within a large audio collection
- Instances of specific audio events

## Database Schema

vggish_to_timescaledb.py creates a `audio_embeddings` table with the following structure:

```sql
CREATE TABLE audio_embeddings (
    timestamp TIMESTAMPTZ NOT NULL,
    audio_file TEXT NOT NULL,
    segment_id INTEGER NOT NULL,
    embedding vector(128),
    metadata JSONB,
    PRIMARY KEY (timestamp, audio_file, segment_id)
);
```

The `metadata` field contains JSON with additional information:
- `segment_duration_s`: Duration of the audio segment
- `segment_start_time_s`: Start time within the original file
- `embedding_dimension`: Size of the embedding vector (128)
- `sample_rate`: Sample rate of the original audio

## Example Queries

### Find the 10 Most Similar Audio Segments

```sql
-- Find audio similar to a specific embedding, place the actual embedding as the array below
SELECT 
    audio_file,
    segment_id,
    timestamp,
    1 - (embedding <=> '[0.1, 0.2, ..., 0.3]'::vector) AS similarity_score
FROM 
    audio_embeddings
ORDER BY 
    embedding <=> '[0.1, 0.2, ..., 0.3]'::vector
LIMIT 10;
```

### Find Segments Similar to an Existing Audio Segment

```sql
-- Find segments similar to an existing segment
WITH target_embedding AS (
    SELECT embedding
    FROM audio_embeddings
    WHERE audio_file = 'your_audio_file.wav' AND segment_id = 5
)
SELECT 
    ae.audio_file,
    ae.segment_id,
    ae.timestamp,
    1 - (ae.embedding <=> te.embedding) AS similarity_score
FROM 
    audio_embeddings ae,
    target_embedding te
WHERE
    ae.audio_file != 'your_audio_file.wav' OR ae.segment_id != 5
ORDER BY 
    ae.embedding <=> te.embedding
LIMIT 10;
```

### Find Segments Within a Specific Time Range

```sql
-- Find similar segments within a time range, place the actual embedding as the array below
SELECT 
    audio_file,
    segment_id,
    timestamp,
    1 - (embedding <=> '[0.1, 0.2, ..., 0.3]'::vector) AS similarity_score
FROM 
    audio_embeddings
WHERE 
    timestamp BETWEEN '2025-05-01' AND '2025-05-15'
ORDER BY 
    embedding <=> '[0.1, 0.2, ..., 0.3]'::vector
LIMIT 10;
```

## Performance Considerations


1. **Index Tuning**: Configure the DISKANN index parameters based on your dataset size and search requirements:
   ```sql
   CREATE INDEX audio_embeddings_embedding_idx 
   ON audio_embeddings USING diskann (embedding vector_cosine_ops)
   WITH (ef_construction = 128, m = 16, ef_search = 64);
   ```
   
   Parameters to tune:
   - `ef_construction`: Higher values increase build time but enables more accurate search results (64-512)
   - `m`: Maximum number of connections per node (8-64)
   - `ef_search`: Controls the accuracy vs. speed tradeoff for queries (higher = more accurate but slower). This parameter specifies the size of the dynamic candidate list used during search. Defaults to 40. Higher values improve query accuracy while making the query slower

2. **Chunking**: Process large collections in smaller batches to manage memory usage.

3. **Hypertable Tuning**: Customize the chunk size based on your query patterns:
   ```sql
   SELECT set_chunk_time_interval('audio_embeddings', INTERVAL '1 day');
   ```

## Advanced Usage

### Processing Different Audio Formats

For non-WAV formats, use `ffmpeg` to convert first:

```bash
# Convert MP3 to WAV
ffmpeg -i input.mp3 -acodec pcm_s16le -ar 44100 -ac 1 output.wav
```

## References

- [VGGish Paper](https://research.google/pubs/pub45611/)
- [TimescaleDB Vector Documentation](https://docs.timescale.com/use-timescale/latest/extensions/vector/)
- [TensorFlow Hub VGGish](https://tfhub.dev/google/vggish/1)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

##  Acknowledgments


- TimescaleDB for the vector extension capabilities
- Google's AudioSet team for the VGGish model
- TensorFlow team for TensorFlow Hub

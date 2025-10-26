import os
import wave


# Directory containing packet_*.pcm files
path = '/Users/vs/Coding/janus-ai/backend/app/uploads/packets-sess_eee7fb10-20251026T093842Z'


def combine_first_n_pcm(
    input_dir: str,
    output_filename: str = 'test_combined.wav',
    limit: int = 50,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width_bytes: int = 2,
) -> str:
    """Combine the first N PCM16 chunks into a single WAV file.

    Args:
        input_dir: Directory with .pcm chunks saved in filename order.
        output_filename: Name of the output WAV file to create.
        limit: Max number of chunks to combine.
        sample_rate: Sample rate in Hz (must match chunks).
        channels: Number of channels (1 for mono).
        sample_width_bytes: Bytes per sample (2 for PCM16).

    Returns:
        Absolute path to the created WAV file.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pcm_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pcm')]
    if not pcm_files:
        raise FileNotFoundError(f"No .pcm files found in: {input_dir}")

    pcm_files.sort()  # rely on lexicographic order (packet_000001..., etc.)
    selected_files = pcm_files[:limit]

    out_path = os.path.join(input_dir, output_filename)

    with wave.open(out_path, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate)

        total_bytes = 0
        for name in selected_files:
            p = os.path.join(input_dir, name)
            with open(p, 'rb') as f:
                data = f.read()
            if not data:
                continue
            wav.writeframes(data)
            total_bytes += len(data)

    print(f"Wrote {len(selected_files)} chunks -> {out_path} ({total_bytes} bytes of PCM)")
    return os.path.abspath(out_path)


if __name__ == '__main__':
    combine_first_n_pcm(path)
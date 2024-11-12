import hashlib
import struct

def simplified_mrsh_v2(data, block_size=4096, window_size=7):
    if len(data) < block_size:
        return None

    def rolling_hash(window):
        return sum(struct.unpack('B', bytes([b]))[0] * (2 ** i) for i, b in enumerate(window)) & 0xFFFFFFFF

    hashes = []
    for i in range(0, len(data) - block_size, block_size // 2):
        block = data[i:i+block_size]
        window_hashes = []
        
        for j in range(len(block) - window_size + 1):
            window = block[j:j+window_size]
            r_hash = rolling_hash(window)
            if r_hash % 2048 == 0:  # Trigger value, can be adjusted
                window_hashes.append(hashlib.md5(window).digest())
        
        if window_hashes:
            block_hash = hashlib.md5(b''.join(window_hashes)).hexdigest()
            hashes.append(block_hash)

    return ':'.join(hashes)

def compare_simplified_mrsh_v2(hash1, hash2):
    if hash1 is None or hash2 is None:
        return 0

    blocks1 = hash1.split(':')
    blocks2 = hash2.split(':')
    
    matches = sum(1 for b1 in blocks1 if b1 in blocks2)
    total_blocks = max(len(blocks1), len(blocks2))
    
    return (matches / total_blocks) * 100

def compare_audio_files_mrsh_v2(file1_path, file2_path):
    try:
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()

        hash1 = simplified_mrsh_v2(data1)
        hash2 = simplified_mrsh_v2(data2)

        similarity_score = compare_simplified_mrsh_v2(hash1, hash2)
        return float(similarity_score), "Simplified MRSH-V2"
    except Exception as e:
        return None, f"Error comparing files with Simplified MRSH-V2: {str(e)}"
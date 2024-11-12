from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import ppdeep
import hashlib  
from Levenshtein import distance as levenshtein_distance
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import struct
import tlsh
from simplified_mrsh_v2 import compare_audio_files_mrsh_v2
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Increase max content length to 500 MB for all files
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'mp3', 'wav', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension in ['.txt', '.pdf', '.doc', '.docx']:
        return 'text'
    elif extension in ['.mp3', '.wav']:
        return 'audio'
    elif extension in ['.mp4', '.avi', '.mov']:
        return 'video'
    else:
        return 'unknown'

def compare_files(file1_path, file2_path, method):
    file_type1 = get_file_type(file1_path)
    file_type2 = get_file_type(file2_path)

    if file_type1 != file_type2:
        return None, "Files are of different types and cannot be compared."

    if file_type1 == 'text':
        if method == 'ssdeep':
            return compare_text_files_ssdeep(file1_path, file2_path)
        elif method == 'levenshtein':
            score, message = compare_text_files_levenshtein(file1_path, file2_path)
            if score is None:
                return None, message
            return score, "Levenshtein"
        else:
            return None, "Unsupported comparison method for text files."
    elif file_type1 == 'audio':
        if method == 'mfcc':
            return compare_audio_files_mfcc(file1_path, file2_path)
        elif method == 'tlsh':
            return compare_audio_files_tlsh(file1_path, file2_path)
        elif method == 'mrsh-v2':
            return compare_audio_files_mrsh_v2(file1_path, file2_path)
        else:
            return None, f"Unsupported comparison method for audio files."
    elif file_type1 == 'video':
        if method == 'video_ssim_hist':
            return compare_video_files(file1_path, file2_path)
        else:
            return None, f"Unsupported comparison method for video files."
    else:
        return None, "Unsupported file type for comparison."

def compare_audio_files_mfcc(file1_path, file2_path):
    try:
        # Load audio files
        y1, sr1 = librosa.load(file1_path, sr=None, mono=True)
        y2, sr2 = librosa.load(file2_path, sr=None, mono=True)

        # Resample to a common sample rate if necessary
        if sr1 != sr2:
            target_sr = min(sr1, sr2)
            y1 = librosa.resample(y1, sr1, target_sr)
            y2 = librosa.resample(y2, sr2, target_sr)
        else:
            target_sr = sr1

        # Extract MFCCs
        mfcc1 = librosa.feature.mfcc(y=y1, sr=target_sr)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=target_sr)

        # Pad or truncate MFCCs to have the same length
        max_length = max(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = np.pad(mfcc1, ((0, 0), (0, max_length - mfcc1.shape[1])), mode='constant')
        mfcc2 = np.pad(mfcc2, ((0, 0), (0, max_length - mfcc2.shape[1])), mode='constant')

        # Flatten MFCCs
        mfcc1_flat = mfcc1.flatten()
        mfcc2_flat = mfcc2.flatten()

        # Calculate cosine similarity
        similarity = 1 - cosine(mfcc1_flat, mfcc2_flat)
        similarity_score = float(similarity * 100)  # Convert to regular Python float

        return similarity_score, "MFCC Cosine Similarity"
    except Exception as e:
        return None, f"Error comparing audio files: {str(e)}"

def compare_audio_files_tlsh(file1_path, file2_path):
    try:
        # Load audio files
        y1, sr1 = librosa.load(file1_path, sr=None, mono=True)
        y2, sr2 = librosa.load(file2_path, sr=None, mono=True)

        # Resample to a common sample rate if necessary
        if sr1 != sr2:
            target_sr = min(sr1, sr2)
            y1 = librosa.resample(y1, sr1, target_sr)
            y2 = librosa.resample(y2, sr2, target_sr)

        # Convert audio to bytes
        bytes1 = y1.tobytes()
        bytes2 = y2.tobytes()

        # Compute TLSH hashes
        hash1 = tlsh.hash(bytes1)
        hash2 = tlsh.hash(bytes2)

        # Compare the hashes
        if hash1 == "TNULL" or hash2 == "TNULL":
            return None, "TLSH hash could not be computed for one or both files"

        distance = tlsh.diff(hash1, hash2)

        # Convert distance to similarity score (0-100 scale)
        max_distance = 300  # This is an arbitrary value, adjust as needed
        similarity_score = max(0, 100 - (distance / max_distance * 100))

        return float(similarity_score), "TLSH"
    except Exception as e:
        return None, f"Error comparing audio files with TLSH: {str(e)}"

def compare_text_files_ssdeep(file1_path, file2_path):
    hash1 = ppdeep.hash_from_file(file1_path)
    hash2 = ppdeep.hash_from_file(file2_path)
    similarity_score = ppdeep.compare(hash1, hash2)
    return similarity_score, "SSDeep"

def compare_text_files_levenshtein(file1_path, file2_path):
    def read_file(file_path):
        encodings = ['utf-8', 'latin-1', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode the file {file_path} with the attempted encodings.")

    try:
        text1 = read_file(file1_path)
        text2 = read_file(file2_path)
        max_length = max(len(text1), len(text2))
        similarity_score = (1 - levenshtein_distance(text1, text2) / max_length) * 100
        return similarity_score, "Levenshtein"
    except ValueError as e:
        return None, str(e)

def compare_audio_files_mrsh_v2(file1_path, file2_path):
    return compare_audio_files_mrsh_v2(file1_path, file2_path)

def compare_video_files(file1_path, file2_path):
    try:
        # Open the video files
        cap1 = cv2.VideoCapture(file1_path)
        cap2 = cv2.VideoCapture(file2_path)

        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the number of frames to sample (e.g., 20 frames)
        sample_count = 20
        step1 = max(1, frame_count1 // sample_count)
        step2 = max(1, frame_count2 // sample_count)

        ssim_scores = []
        histogram_similarities = []

        # Process video frames
        for i in range(0, min(frame_count1, frame_count2), max(step1, step2)):
            cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
            
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if ret1 and ret2:
                # Resize frames to a common size
                frame1 = cv2.resize(frame1, (300, 300))
                frame2 = cv2.resize(frame2, (300, 300))

                # Convert frames to grayscale
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # Compute SSIM
                ssim_score = ssim(gray1, gray2)
                ssim_scores.append(ssim_score)

                # Compute color histograms
                hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

                # Normalize histograms
                cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

                # Compare histograms
                hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                histogram_similarities.append(hist_similarity)

        # Release video captures
        cap1.release()
        cap2.release()

        # Calculate final similarity score
        avg_ssim = np.mean(ssim_scores)
        avg_hist_similarity = np.mean(histogram_similarities)

        # Combine SSIM and histogram similarity (you can adjust weights)
        final_similarity = (avg_ssim * 0.6 + avg_hist_similarity * 0.4) * 100

        return float(final_similarity), "Video Similarity (SSIM + Histogram)"
    except Exception as e:
        return None, f"Error comparing video files: {str(e)}"

def generate_insights(similarity_score, file_type):
    if similarity_score == 100:
        return f"The {file_type} files are identical."
    elif similarity_score > 90:
        return f"The {file_type} files are very similar, with only minor differences."
    elif similarity_score > 70:
        return f"The {file_type} files have significant similarities, but also notable differences."
    elif similarity_score > 50:
        return f"The {file_type} files have some similarities, but are largely different."
    else:
        return f"The {file_type} files are mostly different, with only minor similarities."

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files or 'file-type' not in request.form:
            return jsonify({'error': 'Both files and file type are required'}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        selected_file_type = request.form['file-type']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        
        file_type1 = get_file_type(filename1)
        file_type2 = get_file_type(filename2)
        
        if file_type1 != selected_file_type or file_type2 != selected_file_type:
            return jsonify({'error': f'Selected file type ({selected_file_type}) does not match uploaded file types ({file_type1}, {file_type2})'}), 400
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        try:
            file1.save(file1_path)
            file2.save(file2_path)
            
            # Choose algorithm based on file type
            if selected_file_type == 'audio':
                similarity_score, algorithm = compare_audio_files_mfcc(file1_path, file2_path)
            elif selected_file_type == 'video':
                similarity_score, algorithm = compare_video_files(file1_path, file2_path)
            elif selected_file_type == 'text':
                similarity_score, algorithm = compare_text_files_ssdeep(file1_path, file2_path)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
            
            if similarity_score is None:
                return jsonify({'error': algorithm}), 400

            # Generate report
            report = {
                'file1': filename1,
                'file2': filename2,
                'file_type': selected_file_type,
                'algorithm': algorithm,
                'similarity_score': round(float(similarity_score), 2) if similarity_score is not None else None,
                'insights': generate_insights(float(similarity_score), selected_file_type) if similarity_score is not None else "Unable to generate insights due to comparison error."
            }
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded files
            for path in [file1_path, file2_path]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"Error removing file {path}: {e}")
        
        return jsonify(report)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
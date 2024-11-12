import React, { useState } from 'react';
import ReactDOM from 'react-dom';

function App() {
    const [file1, setFile1] = useState(null);
    const [file2, setFile2] = useState(null);
    const [algorithm, setAlgorithm] = useState('ssdeep');
    const [result, setResult] = useState(null);

    const handleFile1Change = (e) => setFile1(e.target.files[0]);
    const handleFile2Change = (e) => setFile2(e.target.files[0]);
    const handleAlgorithmChange = (e) => setAlgorithm(e.target.value);

    const handleSubmit = async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file1', file1);
        formData.append('file2', file2);
        formData.append('algorithm', algorithm);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        setResult(data.similarity);
    };

    return (
        <div>
            <h1>Similarity Checker</h1>
            <form onSubmit={handleSubmit}>
                <input type="file" onChange={handleFile1Change} required />
                <input type="file" onChange={handleFile2Change} required />
                <select value={algorithm} onChange={handleAlgorithmChange}>
                    <option value="ssdeep">SSDeep</option>
                    <option value="sdhash">SDHash</option>
                    <option value="mrsh-v2">MRSH-V2</option>
                    <option value="tlsh">TLSH</option>
                </select>
                <button type="submit">Check Similarity</button>
            </form>
            {result && <p>Similarity Score: {result}</p>}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));

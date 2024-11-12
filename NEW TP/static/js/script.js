document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        resultDiv.innerHTML = '<p>Processing... Please wait.</p>';
        
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                resultDiv.innerHTML = `
                    <h2>Similarity Report</h2>
                    <p><strong>File 1:</strong> ${data.file1}</p>
                    <p><strong>File 2:</strong> ${data.file2}</p>
                    <p><strong>File Type:</strong> ${data.file_type}</p>
                    <p><strong>Algorithm:</strong> ${data.algorithm}</p>
                    <p><strong>Similarity Score:</strong> ${data.similarity_score}%</p>
                    <p><strong>Insights:</strong> ${data.insights}</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.innerHTML = '<p class="error">An error occurred. Please try again.</p>';
        });
    });
});
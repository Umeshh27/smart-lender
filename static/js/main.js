document.addEventListener('DOMContentLoaded', () => {
    // 1. Tab Switching Logic
    const tabs = document.querySelectorAll('.tab');
    const tabPanes = document.querySelectorAll('.tab-pane');

    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const target = tab.dataset.tab;

                tabs.forEach(t => t.classList.remove('active'));
                tabPanes.forEach(p => p.classList.remove('active'));

                tab.classList.add('active');
                const targetPane = document.getElementById(target);
                if (targetPane) {
                    targetPane.classList.add('active');
                }
            });
        });
    }

    // 2. Drag & Drop File Upload
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    if (uploadZone && fileInput) {
        // Trigger click on file input
        uploadZone.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadZone.classList.remove('dragover');
            }, false);
        });

        // Drop file
        uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileUpload(files[0]);
            }
        });

        // File selection
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileUpload(fileInput.files[0]);
            }
        });
    }

    // 3. File Upload Handler
    function handleFileUpload(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a valid CSV file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Show spinner & hide previous results
        const loadingContainer = document.getElementById('loadingContainer');
        const resultsContainer = document.getElementById('batchResultsContainer');
        const uploadTextElement = uploadZone.querySelector('.upload-text');

        if (loadingContainer) loadingContainer.style.display = 'block';
        if (resultsContainer) resultsContainer.style.display = 'none';
        if (uploadTextElement) uploadTextElement.textContent = `Selected: ${file.name}`;

        fetch('/batch-predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (loadingContainer) loadingContainer.style.display = 'none';
            
            if (data.error) {
                alert(`Error: ${data.error}`);
                if (uploadTextElement) uploadTextElement.textContent = "Choose a file or drag it here";
                return;
            }

            // Populate Batch Summary Statistics
            document.getElementById('totalCount').textContent = data.total;
            document.getElementById('approvedCount').textContent = data.approved;
            document.getElementById('rejectedCount').textContent = data.rejected;
            document.getElementById('approvalRate').textContent = data.approval_rate;

            // Populate Table
            const tbody = document.getElementById('resultsTableBody');
            if (tbody) {
                tbody.innerHTML = '';
                data.results.forEach(row => {
                    const tr = document.createElement('tr');
                    
                    const decisionClass = row.Prediction === 'Approved' ? 'text-success' : 'text-danger';
                    const creditHistoryText = row.Credit_History === 1.0 || row.Credit_History === '1.0' || row.Credit_History === 1 ? 'Good' : 'Bad';
                    
                    tr.innerHTML = `
                        <td>₹${parseFloat(row.ApplicantIncome).toFixed(0)}</td>
                        <td>₹${parseFloat(row.LoanAmount).toFixed(0)}K</td>
                        <td>${creditHistoryText}</td>
                        <td class="${decisionClass}">${row.Prediction}</td>
                        <td>${row.Confidence}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            // Set download URL
            const downloadBtn = document.getElementById('downloadResultsBtn');
            if (downloadBtn) {
                downloadBtn.href = data.download_url;
                downloadBtn.style.display = 'inline-flex';
            }

            if (resultsContainer) resultsContainer.style.display = 'block';
        })
        .catch(error => {
            if (loadingContainer) loadingContainer.style.display = 'none';
            alert(`Network error during batch evaluation: ${error.message}`);
            if (uploadTextElement) uploadTextElement.textContent = "Choose a file or drag it here";
        });
    }

    // 4. Form validation for Single Prediction
    const loanForm = document.getElementById('loanForm');
    if (loanForm) {
        loanForm.addEventListener('submit', (e) => {
            const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value);
            const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value);
            const loanAmount = parseFloat(document.getElementById('LoanAmount').value);
            const loanTerm = parseFloat(document.getElementById('Loan_Amount_Term').value);

            if (isNaN(applicantIncome) || applicantIncome < 0) {
                alert('Applicant Income must be a positive number.');
                e.preventDefault();
                return;
            }
            if (isNaN(coapplicantIncome) || coapplicantIncome < 0) {
                alert('Coapplicant Income must be a positive number.');
                e.preventDefault();
                return;
            }
            if (isNaN(loanAmount) || loanAmount <= 0) {
                alert('Loan Amount must be greater than zero.');
                e.preventDefault();
                return;
            }
            if (isNaN(loanTerm) || loanTerm <= 0) {
                alert('Loan Amount Term must be greater than zero.');
                e.preventDefault();
                return;
            }
        });
    }
});

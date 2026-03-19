document.addEventListener('DOMContentLoaded', () => {

    /* ===== Theme Management ===== */
    const themeBtn = document.getElementById('theme-toggle');
    const themeIcon = themeBtn.querySelector('i');
    const body = document.body;

    if (localStorage.getItem('theme') === 'dark') {
        body.classList.replace('light-mode', 'dark-mode');
        themeIcon.className = 'fa-solid fa-sun';
    }

    themeBtn.addEventListener('click', () => {
        const isLight = body.classList.contains('light-mode');
        body.classList.replace(isLight ? 'light-mode' : 'dark-mode', isLight ? 'dark-mode' : 'light-mode');
        themeIcon.className = isLight ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
        localStorage.setItem('theme', isLight ? 'dark' : 'light');
    });

    /* ===== Particle Generation (Floating Orbs) ===== */
    const particlesContainer = document.getElementById('particles');
    for (let i = 0; i < 15; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = Math.random() * 8 + 4; // 4px to 12px
        p.style.width = `${size}px`;
        p.style.height = `${size}px`;
        p.style.left = `${Math.random() * 100}vw`;
        p.style.animationDuration = `${10 + Math.random() * 15}s`;
        p.style.animationDelay = `${Math.random() * 5}s`;
        particlesContainer.appendChild(p);
    }

    /* ===== DOM Elements ===== */
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const placeholder = document.getElementById('upload-placeholder');
    const previewWrapper = document.getElementById('preview-wrapper');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const generateBtn = document.getElementById('generate-btn');
    const scanLine = document.getElementById('scan-line');
    
    const resultsArea = document.getElementById('results-area');
    const captionCards = document.getElementById('caption-cards');

    let currentFile = null;
    const API_URL = '/predict';

    /* ===== Toast System ===== */
    const toastContainer = document.getElementById('toast-container');
    function showToast(msg, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icon = type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation';
        toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${msg}</span>`;
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'toastOut 0.3s forwards cubic-bezier(0.16,1,0.3,1)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    /* ===== Drag & Drop Logic ===== */
    placeholder.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-active'), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-active'), false);
    });

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files), false);
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (!files.length) return;
        const file = files[0];
        
        if (!file.type.match('image.*')) {
            return showToast('Upload a valid image format.', 'error');
        }
        if (file.size > 5 * 1024 * 1024) {
             return showToast('Image must be under 5 MB.', 'error');
        }

        currentFile = file;
        const reader = new FileReader();

        reader.onload = e => {
            previewImg.src = e.target.result;
            placeholder.style.display = 'none';
            previewWrapper.style.display = 'block';
            generateBtn.disabled = false;
            resultsArea.style.display = 'none'; // hide previous results
            showToast('Image uploaded successfully! Ready to generate.', 'success');
        };

        reader.readAsDataURL(file);
    }

    /* ===== Reset State ===== */
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        placeholder.style.display = 'flex';
        previewWrapper.style.display = 'none';
        generateBtn.disabled = true;
        resultsArea.style.display = 'none';
    });

    /* ===== API Generation ===== */
    generateBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Visual Loading State
        generateBtn.disabled = true;
        generateBtn.classList.add('loading');
        generateBtn.querySelector('.btn-text').textContent = 'Generating...';
        const icon = generateBtn.querySelector('i');
        icon.className = 'fa-solid fa-spinner fa-spin';
        scanLine.style.display = 'block';
        resultsArea.style.display = 'none';

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to generate captions.');
            }

            const data = await response.json();
            renderCaptions(data.captions);

        } catch (error) {
            showToast(error.message === 'Failed to fetch' 
                ? 'Cannot connect. Is the Flask backend running?' 
                : error.message, 'error');
        } finally {
            // Revert Button UI
            generateBtn.classList.remove('loading');
            generateBtn.querySelector('.btn-text').textContent = 'Generate Captions';
            icon.className = 'fa-solid fa-bolt';
            scanLine.style.display = 'none';
            generateBtn.disabled = false;
        }
    });

    /* ===== Render Results ===== */
    function renderCaptions(captionsArr) {
        captionCards.innerHTML = '';
        const captions = Array.isArray(captionsArr) ? captionsArr : [captionsArr];

        captions.forEach((cap, index) => {
            const card = document.createElement('div');
            card.className = 'caption-card';
            card.innerHTML = `
                <p>${cap}</p>
                <button class="copy-btn" title="Copy to clipboard">
                    <i class="fa-regular fa-copy"></i>
                </button>
            `;
            
            // Copy logic
            card.addEventListener('click', () => {
                navigator.clipboard.writeText(cap).then(() => {
                    showToast('Caption copied!', 'success');
                    const copyIcon = card.querySelector('i');
                    copyIcon.className = 'fa-solid fa-check';
                    card.style.borderColor = 'var(--accent-primary)';
                    setTimeout(() => {
                        copyIcon.className = 'fa-regular fa-copy';
                        card.style.borderColor = 'var(--card-border)';
                    }, 2000);
                });
            });

            captionCards.appendChild(card);
            
            // Stagger animation
            setTimeout(() => {
                card.classList.add('show');
            }, index * 150 + 100);
        });

        resultsArea.style.display = 'block';
        resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});

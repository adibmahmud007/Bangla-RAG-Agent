from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your RAG system
from app.rag_model import EnhancedRAGSystemWithGroq

# Create FastAPI app
app = FastAPI(
    title="Bengali RAG System",
    description="A Bengali RAG System with FastAPI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
print("🚀 Initializing Bengali RAG System...")
rag_system = EnhancedRAGSystemWithGroq(vector_store_path='app/vector_store.pkl')
print("✅ RAG System initialized successfully")

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    answer: str
    confidence: float
    keyword_matches: float
    semantic_score: float
    matching_keywords: list
    full_chunk: str
    model: str
    success: bool
    error: str = None

class StatusResponse(BaseModel):
    status: str
    chunks_loaded: int = None
    groq_available: bool = None
    model: str = None
    error: str = None

# Premium White Theme HTML Template with Side Panel
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bengali RAG System - Premium</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #64748b 100%);
            position: relative;
            min-height: 100vh;
            color: #ffffff;
            overflow-x: hidden;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .container {
            max-width: 950px;
            margin: 0 auto;
            padding: 25px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }

        .header {
            text-align: center;
            padding: 40px 30px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            text-shadow: 0 2px 10px rgba(59, 130, 246, 0.2);
        }

        .header p {
            color: #64748b;
            font-size: 1.2rem;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 30px;
            position: relative;
        }

        .question-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 35px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        .question-input {
            width: 100%;
            background: rgba(248, 250, 252, 0.8);
            border: 2px solid rgba(148, 163, 184, 0.3);
            border-radius: 18px;
            padding: 25px;
            font-size: 18px;
            color: #334155;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
            min-height: 130px;
            font-family: inherit;
            line-height: 1.6;
        }

        .question-input::placeholder {
            color: #94a3b8;
            font-weight: 400;
        }

        .question-input:focus {
            border-color: #3b82f6;
            box-shadow: 
                0 0 25px rgba(59, 130, 246, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            background: rgba(255, 255, 255, 0.9);
        }

        .ask-button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
            color: #ffffff;
            border: none;
            border-radius: 16px;
            padding: 18px 45px;
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 25px;
            transition: all 0.3s ease;
            box-shadow: 
                0 12px 30px rgba(59, 130, 246, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            text-transform: uppercase;
            letter-spacing: 1px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .ask-button:hover {
            transform: translateY(-3px);
            box-shadow: 
                0 18px 40px rgba(59, 130, 246, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
        }

        .ask-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .response-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 35px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            min-height: 250px;
            display: none;
        }

        .response-container.show {
            display: block;
            animation: fadeInUp 0.5s ease;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .answer-text {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 18px;
            padding: 30px;
            font-size: 20px;
            line-height: 1.7;
            color: #1e293b;
            margin-bottom: 30px;
            font-weight: 500;
            box-shadow: 
                0 8px 25px rgba(34, 197, 94, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        .chunk-text {
            background: rgba(248, 250, 252, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 25px;
            font-size: 15px;
            line-height: 1.7;
            color: #475569;
            max-height: 320px;
            overflow-y: auto;
            font-weight: 400;
        }

        .details-button {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6366f1 100%);
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 14px 28px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
            box-shadow: 
                0 8px 20px rgba(139, 92, 246, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .details-button:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 12px 25px rgba(139, 92, 246, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.15);
        }

        /* Side Panel for Details */
        .details-panel {
            position: fixed;
            top: 0;
            right: -400px;
            width: 400px;
            height: 100vh;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-left: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow: -10px 0 30px rgba(0, 0, 0, 0.2);
            transition: right 0.4s ease;
            z-index: 1000;
            overflow-y: auto;
            padding: 30px;
        }

        .details-panel.show {
            right: 0;
        }

        .details-panel-header {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 25px;
            color: #1e293b;
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        }

        .panel-close-button {
            position: absolute;
            top: 20px;
            right: 25px;
            background: rgba(148, 163, 184, 0.1);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            color: #64748b;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .panel-close-button:hover {
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: #ef4444;
            transform: scale(1.1);
        }

        .panel-stats-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }

        .panel-stat-item {
            background: rgba(248, 250, 252, 0.8);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }

        .panel-stat-label {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .panel-stat-value {
            font-size: 24px;
            font-weight: 800;
            color: #3b82f6;
        }

        .panel-keywords-section {
            background: rgba(248, 250, 252, 0.8);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .panel-keywords-list {
            color: #7c3aed;
            font-weight: 600;
            font-size: 14px;
            line-height: 1.6;
        }

        /* Overlay for side panel */
        .panel-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(3px);
            z-index: 999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }

        .panel-overlay.show {
            opacity: 1;
            visibility: visible;
        }

        .loading {
            opacity: 0.6;
            pointer-events: none;
        }

        .empty-state {
            color: #94a3b8;
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
            font-weight: 400;
        }

        .error-state {
            color: #ef4444;
            text-align: center;
            padding: 25px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 15px;
            font-weight: 500;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.6), rgba(99, 102, 241, 0.6));
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.8), rgba(99, 102, 241, 0.8));
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 20px 15px;
            }

            .header h1 {
                font-size: 2.2rem;
            }

            .header p {
                font-size: 1rem;
            }

            .question-input {
                font-size: 16px;
                min-height: 110px;
                padding: 20px;
            }

            .answer-text {
                font-size: 18px;
                padding: 25px;
            }

            .details-panel {
                width: 100%;
                right: -100%;
            }

            .panel-stats-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }

            .panel-stat-value {
                font-size: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multilingual RAG System</h1>
            <p>Premium AI-Powered Question Answering System</p>
        </div>


        <div class="main-content">
            <div class="question-container">
                <textarea 
                    id="questionInput" 
                    class="question-input" 
                    placeholder="প্রশ্ন করুন অথবা যেকোনো কিছু জিজ্ঞাসা করুন..."
                    rows="4"
                ></textarea>
                <button id="askButton" class="ask-button">প্রশ্ন করুন</button>
            </div>

            <div id="responseContainer" class="response-container">
                <div class="section-title">
                    <span>📝</span> উত্তর
                </div>
                <div id="answerText" class="answer-text">
                    <div class="empty-state">এখানে উত্তর দেখা যাবে...</div>
                </div>

                <div class="section-title">
                    <span>📄</span> সম্পূর্ণ প্রসঙ্গ
                </div>
                <div id="chunkText" class="chunk-text">
                    <div class="empty-state">এখানে সম্পূর্ণ প্রসঙ্গ দেখা যাবে...</div>
                </div>

                <button id="detailsButton" class="details-button">📊 বিস্তারিত তথ্য দেখুন</button>
            </div>
        </div>
    </div>

    <!-- Side Panel for Details -->
    <div id="panelOverlay" class="panel-overlay"></div>
    <div id="detailsPanel" class="details-panel">
        <button id="panelCloseButton" class="panel-close-button">&times;</button>
        <div class="details-panel-header">📊 বিস্তারিত বিশ্লেষণ</div>
        
        <div class="panel-stats-grid">
            <div class="panel-stat-item">
                <div class="panel-stat-label">📊 Confidence</div>
                <div class="panel-stat-value" id="panelConfidenceValue">-</div>
            </div>
            <div class="panel-stat-item">
                <div class="panel-stat-label">🔢 Keyword Matches</div>
                <div class="panel-stat-value" id="panelMatchesValue">-</div>
            </div>
            <div class="panel-stat-item">
                <div class="panel-stat-label">🎯 Semantic Score</div>
                <div class="panel-stat-value" id="panelSemanticValue">-</div>
            </div>
            <div class="panel-stat-item">
                <div class="panel-stat-label">🤖 AI Model</div>
                <div class="panel-stat-value" id="panelModelValue">-</div>
            </div>
        </div>
        
        <div class="panel-keywords-section">
            <div class="panel-stat-label">🔍 মিলেছে এই শব্দগুলো</div>
            <div class="panel-keywords-list" id="panelKeywordsValue">-</div>
        </div>
    </div>

    <script>
        const questionInput = document.getElementById('questionInput');
        const askButton = document.getElementById('askButton');
        const responseContainer = document.getElementById('responseContainer');
        const answerText = document.getElementById('answerText');
        const chunkText = document.getElementById('chunkText');
        const detailsButton = document.getElementById('detailsButton');
        const detailsPanel = document.getElementById('detailsPanel');
        const panelOverlay = document.getElementById('panelOverlay');
        const panelCloseButton = document.getElementById('panelCloseButton');
        const panelConfidenceValue = document.getElementById('panelConfidenceValue');
        const panelMatchesValue = document.getElementById('panelMatchesValue');
        const panelSemanticValue = document.getElementById('panelSemanticValue');
        const panelModelValue = document.getElementById('panelModelValue');
        const panelKeywordsValue = document.getElementById('panelKeywordsValue');

        let currentData = null;

        async function handleQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            // Set loading state
            askButton.disabled = true;
            askButton.textContent = 'প্রক্রিয়াকরণ...';
            
            // Show response container
            responseContainer.classList.add('show');
            
            // Clear previous results
            answerText.innerHTML = '<div class="empty-state">উত্তর তৈরি হচ্ছে...</div>';
            chunkText.innerHTML = '<div class="empty-state">চাংক লোড হচ্ছে...</div>';

            try {
                // Call your RAG API
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: question })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                currentData = data;
                
                // Update answer
                answerText.innerHTML = data.answer || 'উত্তর পাওয়া যায়নি';
                
                // Update chunk
                chunkText.innerHTML = data.full_chunk || 'চাংক পাওয়া যায়নি';
                
            } catch (error) {
                console.error('Error:', error);
                answerText.innerHTML = '<div class="error-state">ত্রুটি ঘটেছে। আবার চেষ্টা করুন।</div>';
                chunkText.innerHTML = '<div class="error-state">ডেটা লোড করতে সমস্যা হয়েছে।</div>';
            }
            
            // Reset button
            askButton.disabled = false;
            askButton.textContent = 'প্রশ্ন করুন';
        }

        function showDetails() {
            if (!currentData) return;
            
            // Update panel values
            panelConfidenceValue.textContent = (currentData.confidence || 0).toFixed(3);
            panelMatchesValue.textContent = (currentData.keyword_matches || 0).toFixed(1);
            panelSemanticValue.textContent = (currentData.semantic_score || 0).toFixed(3);
            panelModelValue.textContent = currentData.model || 'llama-3.3-70b-versatile';
            panelKeywordsValue.textContent = (currentData.matching_keywords || []).join(', ') || 'কোনো কীওয়ার্ড পাওয়া যায়নি';
            
            // Show panel
            panelOverlay.classList.add('show');
            detailsPanel.classList.add('show');
        }

        function hideDetails() {
            panelOverlay.classList.remove('show');
            detailsPanel.classList.remove('show');
        }

        // Event listeners
        askButton.addEventListener('click', handleQuestion);
        detailsButton.addEventListener('click', showDetails);
        panelCloseButton.addEventListener('click', hideDetails);
        
        // Close panel when clicking overlay
        panelOverlay.addEventListener('click', (e) => {
            if (e.target === panelOverlay) {
                hideDetails();
            }
        });

        // Handle Enter key
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                handleQuestion();
            }
        });

        // Handle Escape key to close panel
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                hideDetails();
            }
        });
    </script>
</body>
</html>'''

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main UI"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Handle search requests from the UI"""
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'No query provided',
                    'answer': 'প্রশ্ন লিখুন',
                    'confidence': 0.0,
                    'keyword_matches': 0.0,
                    'semantic_score': 0.0,
                    'matching_keywords': [],
                    'full_chunk': '',
                    'model': 'Error',
                    'success': False
                }
            )
        
        # Get answer from RAG system
        result = rag_system.search_and_answer(query, top_k=3, use_groq=True)
        
        # Extract semantic score from the best chunk if available
        semantic_score = 0.0
        if result.get('source_chunks') and len(result['source_chunks']) > 0:
            semantic_score = result['source_chunks'][0].get('semantic_score', 0.0)
        
        # Format response
        response = SearchResponse(
            answer=result.get('answer', 'উত্তর পাওয়া যায়নি'),
            confidence=result.get('confidence', 0.0),
            keyword_matches=result.get('keyword_matches', 0.0),
            semantic_score=semantic_score,
            matching_keywords=result.get('matching_keywords', []),
            full_chunk=result.get('full_chunk', ''),
            model='llama-3.3-70b-versatile',
            success=True
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in search endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'answer': 'সিস্টেম ত্রুটি',
                'confidence': 0.0,
                'keyword_matches': 0.0,
                'semantic_score': 0.0,
                'matching_keywords': [],
                'full_chunk': '',
                'model': 'Error',
                'success': False
            }
        )

@app.get("/api/status", response_model=StatusResponse)
async def status():
    """Health check endpoint"""
    try:
        # Check if RAG system is working
        chunks_count = len(rag_system.vector_store['chunks']) if rag_system.vector_store else 0
        
        return StatusResponse(
            status='healthy',
            chunks_loaded=chunks_count,
            groq_available=rag_system.groq_client is not None,
            model='llama-3.3-70b-versatile'
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'error',
                'error': str(e)
            }
        )

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Bengali RAG System FastAPI Server")
    print("="*80)
    print("📡 Server URL: http://localhost:8000")
    print("🌐 Web Interface: http://localhost:8000")
    print("📊 API Endpoint: http://localhost:8000/api/search")
    print("❤️  Status Check: http://localhost:8000/api/status")
    print("📖 API Docs: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("="*80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
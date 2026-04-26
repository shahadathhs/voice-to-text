"""Documentation routes for multiple API documentation viewers."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.core.config import settings

router = APIRouter(tags=["Documentation"])


RAPIDOC_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RapiDoc - Voice-to-Text API</title>
    <script src="https://cdn.jsdelivr.net/npm/rapidoc/dist/rapidoc-min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
        }}
        rapi-doc {{
            height: 100vh;
            width: 100%;
            display: flex;
            flex-direction: column;
        }}
    </style>
</head>
<body>
    <rapi-doc
        spec-url="/openapi.json"
        theme="dark"
        bg-color="#1a1a1a"
        text-color="#e0e0e0"
        header-color="#2d2d2d"
        primary-color="#3b82f6"
        sort-endpoints-by="path"
        default-schema-tab="example"
        show-info="true"
        allow-server-selection="false"
        allow-authentication="false"
        render-style="read"
        show-components="true"
        use-path-in-nav-bar="true"
        nav-bg-color="#2d2d2d"
        nav-text-color="#e0e0e0"
        nav-hover-bg-color="#3b82f6"
        nav-hover-text-color="#ffffff"
        nav-item-spacing="compact"
        regular-font="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif"
        mono-font="'SF Mono', 'Monaco', 'Andale Mono', 'Ubuntu Mono', monospace"
        font-size="large"
        show-header="true"
        header-style="row"
        layout="row"
        response-area-height="400px">
        <div slot="header" style="display: flex; align-items: center; gap: 16px; padding: 16px;">
            <div style="font-size: 24px; font-weight: bold;">🎙️ Voice-to-Text API</div>
            <div style="flex: 1;"></div>
            <div style="font-size: 14px; color: #9ca3af;">
                AI-powered voice transcription using OpenAI Whisper
            </div>
        </div>
    </rapi-doc>
</body>
</html>
"""


DOCS_HUB_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Hub - Voice-to-Text API</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            width: 100%;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 60px;
        }}

        .header h1 {{
            font-size: 3rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
        }}

        .header p {{
            font-size: 1.25rem;
            opacity: 0.9;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }}

        .card {{
            background: white;
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            display: block;
        }}

        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
        }}

        .card-icon {{
            font-size: 3rem;
            margin-bottom: 16px;
        }}

        .card-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: #1a202c;
        }}

        .card-description {{
            font-size: 1rem;
            color: #4a5568;
            line-height: 1.6;
            margin-bottom: 16px;
        }}

        .card-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }}

        .badge-recommended {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .badge-alternative {{
            background: #e2e8f0;
            color: #4a5568;
        }}

        .features {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        }}

        .features h2 {{
            font-size: 1.75rem;
            margin-bottom: 24px;
            color: #1a202c;
        }}

        .features ul {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
        }}

        .features li {{
            display: flex;
            align-items: center;
            gap: 12px;
            color: #4a5568;
        }}

        .features li::before {{
            content: "✓";
            display: flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            font-size: 0.875rem;
            flex-shrink: 0;
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
        }}

        .footer a {{
            color: white;
            text-decoration: underline;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2rem;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Voice-to-Text API</h1>
            <p>Choose your preferred documentation experience</p>
        </div>

        <div class="grid">
            <a href="/rapidoc" class="card">
                <div class="card-icon">⭐</div>
                <div class="card-title">RapiDoc</div>
                <div class="card-description">
                    Modern, responsive API documentation with a beautiful dark theme.
                    Provides an excellent developer experience with interactive examples.
                </div>
                <span class="card-badge badge-recommended">⭐ Recommended</span>
            </a>

            <a href="/docs" class="card">
                <div class="card-icon">📖</div>
                <div class="card-title">Swagger UI</div>
                <div class="card-description">
                    Classic interactive documentation with Try It Out feature.
                    Widely used and familiar to most developers.
                </div>
                <span class="card-badge badge-alternative">Classic</span>
            </a>

            <a href="/redoc" class="card">
                <div class="card-icon">📚</div>
                <div class="card-title">ReDoc</div>
                <div class="card-description">
                    Beautiful reference documentation with clean layout.
                    Great for reading and sharing API documentation.
                </div>
                <span class="card-badge badge-alternative">Reference</span>
            </a>
        </div>

        <div class="features">
            <h2>🎯 Key Features</h2>
            <ul>
                <li>Multiple Audio Formats: WAV, MP3, OGG, M4A, FLAC, AAC</li>
                <li>Translation: Convert non-English audio to English</li>
                <li>Speaker Diarization: Identify and label different speakers</li>
                <li>Multiple Backends: OpenAI Whisper or Hugging Face</li>
                <li>Fully Local: All processing happens on your machine</li>
                <li>Fast & Responsive: Built with FastAPI and Pydantic</li>
            </ul>
        </div>

        <div class="footer">
            <p>
                <strong>Version:</strong> {settings.app_version} |
                <strong>Environment:</strong> {settings.environment.capitalize()} |
                <a href="https://github.com/shahadathhs/voice-to-text">GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""


@router.get("/rapidoc", response_class=HTMLResponse, include_in_schema=False)
async def rapidoc():
    """RapiDoc documentation viewer with modern dark theme.

    RapiDoc is a modern, responsive API documentation viewer that provides:
    - Beautiful dark theme with excellent contrast
    - Interactive request/response examples
    - Schema examples by default
    - Superior mobile experience
    - Fast loading and smooth interactions
    """
    return RAPIDOC_HTML


@router.get("/docs-hub", response_class=HTMLResponse, include_in_schema=False)
async def docs_hub():
    """Documentation hub - Choose your preferred documentation viewer.

    This landing page provides an overview of all available documentation options:
    - RapiDoc (⭐ Recommended) - Modern and responsive
    - Swagger UI - Classic interactive docs
    - ReDoc - Beautiful reference documentation

    Each viewer offers a different experience, so choose the one that works best for you!
    """
    return DOCS_HUB_HTML

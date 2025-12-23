"""
Response Formatter for Multi-Modal Content
Makes PDF, CSV, and Image analysis results look professional
"""


def format_multimodal_response(
    response: str,
    file_type: str,
    file_name: str,
    metadata: dict = None, # type: ignore
    theme: str = "dark"
) -> str:
    """
    Format multi-modal responses with beautiful styling
    
    Args:
        response: Raw response from Gemini
        file_type: 'image', 'data', or 'pdf'
        file_name: Name of uploaded file
        metadata: File metadata
        theme: 'dark' or 'light'
    
    Returns:
        Beautifully formatted HTML response
    """
    # Theme colors
    if theme == "dark":
        colors = {
            'bg': 'rgba(30, 41, 59, 0.5)',
            'border': 'rgba(99, 165, 250, 0.3)',
            'accent': '#60a5fa',
            'text': '#f1f5f9',
            'text_secondary': '#cbd5e1',
            'section_bg': 'rgba(51, 65, 85, 0.3)',
        }
    else:
        colors = {
            'bg': 'rgba(248, 250, 252, 0.8)',
            'border': 'rgba(37, 99, 235, 0.2)',
            'accent': '#2563eb',
            'text': '#0f172a',
            'text_secondary': '#475569',
            'section_bg': 'rgba(241, 245, 249, 0.5)',
        }
    
    # File type icons and labels
    file_info = {
        'image': {'icon': '📸', 'label': 'Image Analysis'},
        'data': {'icon': '📊', 'label': 'Data Analysis'},
        'pdf': {'icon': '📄', 'label': 'Document Analysis'}
    }
    
    info = file_info.get(file_type, {'icon': '📎', 'label': 'File Analysis'})
    
    # Build metadata display
    meta_parts = []
    if metadata:
        if file_type == 'pdf':
            meta_parts.append(f"{metadata.get('num_pages', '?')} pages")
            meta_parts.append(f"{metadata.get('file_size_mb', 0):.1f} MB")
        elif file_type == 'data':
            shape = metadata.get('shape', (0, 0))
            meta_parts.append(f"{shape[0]} rows × {shape[1]} cols")
        elif file_type == 'image':
            meta_parts.append(f"{metadata.get('width', '?')} × {metadata.get('height', '?')} px")
    
    meta_text = " • ".join(meta_parts) if meta_parts else ""
    
    # Format the response text (add structure)
    formatted_response = structure_response(response)
    
    # Build the final HTML
    html = f"""
<div style="
    background: {colors['bg']};
    border: 1px solid {colors['border']};
    border-left: 4px solid {colors['accent']};
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    backdrop-filter: blur(10px);
">
    <!-- File Header -->
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid {colors['border']};
    ">
        <span style="font-size: 1.5em;">{info['icon']}</span>
        <div style="flex: 1;">
            <div style="
                font-weight: 600;
                font-size: 1.1em;
                color: {colors['accent']};
                margin-bottom: 4px;
            ">{info['label']}</div>
            <div style="
                font-size: 0.9em;
                color: {colors['text_secondary']};
            ">
                <strong>{file_name}</strong>
                {f' • {meta_text}' if meta_text else ''}
            </div>
        </div>
    </div>
    
    <!-- Analysis Content -->
    <div style="
        color: {colors['text']};
        line-height: 1.6;
    ">
        {formatted_response}
    </div>
</div>
"""
    
    return html


def structure_response(text: str) -> str:
    """
    Add structure to plain text response
    
    - Detects sections (##, ###, numbered lists)
    - Adds proper spacing
    - Formats lists and bullets
    """
    import re
    
    # Split into lines
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            formatted_lines.append('<br>')
            continue
        
        # Detect headers
        if line.startswith('## '):
            header = line.replace('## ', '')
            formatted_lines.append(f"""
<div style="
    font-size: 1.2em;
    font-weight: 600;
    color: var(--primary);
    margin-top: 20px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
">{header}</div>
""")
        
        elif line.startswith('### '):
            header = line.replace('### ', '')
            formatted_lines.append(f"""
<div style="
    font-size: 1.1em;
    font-weight: 600;
    margin-top: 16px;
    margin-bottom: 8px;
">{header}</div>
""")
        
        # Detect numbered items
        elif re.match(r'^\d+\.', line):
            formatted_lines.append(f"""
<div style="
    margin: 8px 0;
    padding-left: 8px;
">
    <strong>{line.split('.')[0]}.</strong> {'.'.join(line.split('.')[1:]).strip()}
</div>
""")
        
        # Detect bullet points
        elif line.startswith('- ') or line.startswith('* '):
            content = line[2:].strip()
            formatted_lines.append(f"""
<div style="
    margin: 6px 0;
    padding-left: 20px;
    position: relative;
">
    <span style="
        position: absolute;
        left: 0;
        color: var(--accent);
    ">▸</span>
    {content}
</div>
""")
        
        # Regular paragraph
        else:
            # Check if it's bold (contains **)
            if '**' in line:
                line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            
            formatted_lines.append(f'<div style="margin: 8px 0;">{line}</div>')
    
    return ''.join(formatted_lines)


def add_section_divider(title: str, theme: str = "dark") -> str:
    """Add a visual section divider"""
    color = '#60a5fa' if theme == 'dark' else '#2563eb'
    
    return f"""
<div style="
    margin: 24px 0 16px 0;
    padding: 12px 0;
    border-top: 2px solid {color};
    border-bottom: 1px solid rgba(99, 165, 250, 0.2);
">
    <div style="
        font-size: 1.1em;
        font-weight: 600;
        color: {color};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    ">{title}</div>
</div>
"""


# Quick test
if __name__ == "__main__":
    sample_response = """## Academic Performance

Your transcript shows exceptional performance.

### Key Achievements

1. Overall GPA: 3.92/4.0
2. Multiple distinctions in ML courses
3. Strong foundation in mathematics

### Recommendations

- Continue with advanced ML courses
- Consider research opportunities
- Build portfolio projects"""

    formatted = format_multimodal_response(
        response=sample_response,
        file_type='pdf',
        file_name='transcript.pdf',
        metadata={'num_pages': 2, 'file_size_mb': 0.4},
        theme='dark'
    )
    
    print("✓ Formatter ready!")
    print(f"Sample output length: {len(formatted)} chars")

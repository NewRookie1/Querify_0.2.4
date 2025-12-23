"""
Mode-Aware Response Formatter
Formats responses differently based on agent mode for better presentation
"""

import re

def format_mode_response(response: str, mode: str, theme: str = "dark") -> str:
    """
    Format response based on active mode for optimal presentation in Gradio.
    
    Args:
        response: Raw response from Gemini
        mode: Active mode display name
    
    Returns:
        Formatted Markdown/HTML string
    """
    # Map friendly names to keys
    mode_map = {
        "🤖 Smart Mode (Auto-detect)": "smart",
        "🎓 Teaching Mode (Detailed)": "teaching",
        "⚡ Quick Mode (Concise)": "quick",
        "🔬 Research Mode (Citations)": "research",
        "💻 Code Mode (Technical)": "code",
        "💬 Conversation Mode": "conversation"
    }
    
    mode_key = mode_map.get(mode, "smart")
    
    # Apply mode-specific formatting
    if mode_key == "teaching":
        return format_teaching_response(response)
    elif mode_key == "quick":
        return format_quick_response(response)
    elif mode_key == "code":
        return format_code_response(response)
    elif mode_key == "research":
        return format_research_response(response)
    elif mode_key == "conversation":
        return format_conversation_response(response)
    else:
        return format_smart_response(response)


# ============ MODE FORMATTERS ============ #

def format_teaching_response(response: str) -> str:
    """Teaching mode: Highlighting concepts and analogies"""
    
    # 1. Highlight Analogies (Boxed style)
    response = highlight_analogies(response)
    
    # 2. Colorize Headers for visual hierarchy
    response = colorize_headers(response, color="var(--color-accent)")
    
    # 3. Add separator lines between major sections
    response = add_visual_dividers(response)
    
    return response


def format_quick_response(response: str) -> str:
    """Quick mode: Clean, bold, direct"""
    
    # 1. Bold key terms automatically if they are at start of bullets
    response = re.sub(r'^- ([A-Za-z0-9 ]+):', r'- **\1**:', response, flags=re.MULTILINE)
    
    # 2. Compact headers (Icon only, no colors to keep it simple)
    response = response.replace("## ", "### ⚡ ")
    
    return response.strip()


def format_code_response(response: str) -> str:
    """Code mode: Preserves raw markdown for Gradio's syntax highlighting"""
    
    # CRITICAL: Do NOT manually format code blocks. 
    # Let Gradio handle syntax highlighting and copy buttons.
    
    # Just add a subtle header color
    response = colorize_headers(response, color="#6366f1") # Indigo
    
    return response


def format_research_response(response: str) -> str:
    """Research mode: Academic style"""
    
    # 1. Format Citations [Source X] -> Superscript
    response = re.sub(
        r'\[Source (\d+)\]', 
        r'<sup>**[Source \1]**</sup>', 
        response
    )
    
    # 2. Highlight "Key Findings"
    response = response.replace("**Key Finding**", "🔎 **Key Finding**")
    response = response.replace("**Hypothesis**", "🧪 **Hypothesis**")
    
    # 3. Formal headers
    response = colorize_headers(response, color="#10b981") # Emerald
    
    return response


def format_conversation_response(response: str) -> str:
    """Conversation mode: Minimal changes"""
    # Remove large headers to keep it conversational
    response = response.replace("## ", "**")
    response = response.replace("### ", "**")
    return response.strip()


def format_smart_response(response: str) -> str:
    """Smart mode: Balanced formatting"""
    response = colorize_headers(response, color="var(--color-accent)")
    response = highlight_analogies(response)
    return response


# ============ UTILITIES ============ #

def colorize_headers(text: str, color: str) -> str:
    """
    Injects HTML span colors into Markdown headers without breaking structure.
    """
    # Replace ## Title with ## <span style='color:...'>Title</span>
    # This keeps it as a valid Markdown header for the Table of Contents, but colors it.
    
    def replacer(match):
        level = match.group(1) # ##
        content = match.group(2) # Title
        return f"{level} <span style='color: {color};'>{content}</span>"
        
    text = re.sub(r'^(#{1,4})\s+(.+)$', replacer, text, flags=re.MULTILINE)
    return text


def highlight_analogies(text: str) -> str:
    """
    Wraps analogy paragraphs in a styled HTML box.
    Uses Gradio standard vars for Light/Dark compatibility.
    """
    analogy_patterns = [
        r'(Imagine .+?)(?=\n\n|\Z)',
        r'(Think of it like .+?)(?=\n\n|\Z)',
        r'(It\'s like .+?)(?=\n\n|\Z)',
        r'(For example.+?)(?=\n\n|\Z)',
        r'(Consider .+?)(?=\n\n|\Z)'
    ]
    
    # CSS that works in both light and dark mode using Gradio variables
    box_style = (
        "background-color: var(--background-fill-secondary); "
        "border-left: 4px solid var(--color-accent); "
        "padding: 12px; "
        "margin: 10px 0; "
        "border-radius: 0 8px 8px 0;"
    )
    
    for pattern in analogy_patterns:
        # We use a lambda to wrap the match in a div
        text = re.sub(
            pattern,
            lambda m: f'<div style="{box_style}">💡 <em>{m.group(1)}</em></div>',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
    
    return text


def add_visual_dividers(text: str) -> str:
    """Adds a subtle HTML divider before Level 2 headers"""
    divider = '<div style="width: 100%; height: 1px; background: var(--border-color-primary); margin: 20px 0;"></div>'
    
    # Insert divider before headers, but not at the very start
    lines = text.split('\n')
    output = []
    for i, line in enumerate(lines):
        if line.startswith('## ') and i > 0:
            output.append(divider)
        output.append(line)
        
    return '\n'.join(output)
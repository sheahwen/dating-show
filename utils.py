import re

def extract_content_without_thoughts(content: str) -> str:
    """
    Extract actual content from model response, removing <think> tags and their contents.
    
    Args:
        content: The full response content that may include <think>...</think> tags
        
    Returns:
        The content with thoughts removed and cleaned up
    """
    # Remove everything between <think> and </think> tags
    content_without_thoughts = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Clean up any extra whitespace at the beginning and end
    return content_without_thoughts.strip()
